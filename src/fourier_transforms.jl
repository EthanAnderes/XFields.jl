#%% Transform types
#%% ============================================================

abstract type Transform end

# nᵢ ≡ number of grid evaluations to a side
# pᵢ ≡ period (e.g. grid evals at {0, pᵢ/nᵢ, pᵢ2/nᵢ, …, pᵢ(nᵢ-1)/nᵢ})
abstract type FourierTransform{nᵢ,pᵢ,d}  <: Transform end
abstract type rFourierTransform{nᵢ,pᵢ,d} <: FourierTransform{nᵢ,pᵢ,d} end
abstract type cFourierTransform{nᵢ,pᵢ,d} <: FourierTransform{nᵢ,pᵢ,d} end

rFT{d} = rFourierTransform{nᵢ,pᵢ,d} where {nᵢ,pᵢ} 
cFT{d} = cFourierTransform{nᵢ,pᵢ,d} where {nᵢ,pᵢ} 

#  This allows broadcasting an fft plan to slices indexed by trailing dimensions
#  note: t isa Int or a tuple of ints
abstract type LastDimSize{t} end



#%% Low level container for forward/backward plans and normalization
#%% -------------------------------------------------------------
#%% Instances of these containers know how to mult and divide


# Here we 
struct rFFTholder{T<:Real,d}
    FT::FFTW.rFFTWPlan{T,-1,false,d}
    IFT::FFTW.rFFTWPlan{Complex{T},1,false,d}
    normalize_FT::T
    normalize_IFT::T
end
struct Adjoint_rFFTholder{T<:Real,d}
    FT::FFTW.rFFTWPlan{T,-1,false,d}
    IFT::FFTW.rFFTWPlan{Complex{T},1,false,d}
    normalize_FT::T
    normalize_IFT::T
end


struct cFFTholder{T<:Real,d} # d is the total dimension of the array it operates on
    FT::FFTW.cFFTWPlan{Complex{T},-1,false,d}
    IFT::FFTW.cFFTWPlan{Complex{T},1,false,d}
    normalize_FT::T
    normalize_IFT::T
end
struct Adjoint_cFFTholder{T<:Real,d} # d is the total dimension of the array it operates on
    FT::FFTW.cFFTWPlan{Complex{T},-1,false,d}
    IFT::FFTW.cFFTWPlan{Complex{T},1,false,d}
    normalize_FT::T
    normalize_IFT::T
end


(*)(p::rFFTholder, x::Array) = p.normalize_FT .* (p.FT * x)
(*)(p::cFFTholder, x::Array) = p.normalize_FT .* (p.FT * x)
(*)(p::Adjoint_rFFTholder, x::Array) = p.normalize_FT .* (p.IFT * x)
(*)(p::Adjoint_cFFTholder, x::Array) = p.normalize_FT .* (p.IFT * x)

(\)(p::rFFTholder, x::Array) = p.normalize_IFT .* (p.IFT * x) 
(\)(p::cFFTholder, x::Array) = p.normalize_IFT .* (p.IFT * x) 
(\)(p::Adjoint_rFFTholder, x::Array) = p.normalize_IFT .* (p.FT * x) 
(\)(p::Adjoint_cFFTholder, x::Array) = p.normalize_IFT .* (p.FT * x) 


adjoint(p::rFFTholder{T,d}) where {T,d} = Adjoint_rFFTholder{T,d}(p.FT,p.IFT,p.normalize_FT,p.normalize_IFT)
adjoint(p::cFFTholder{T,d}) where {T,d} = Adjoint_cFFTholder{T,d}(p.FT,p.IFT,p.normalize_FT,p.normalize_IFT)

transpose(p::rFFTholder) = p
transpose(p::cFFTholder) = p


#%% Plans (TODO: get rid of the allocation for the planned ffts)
#%% -------------------------------------------------------------

@generated function plan(::Type{F}) where {nᵢ, pᵢ, d, F<:rFourierTransform{nᵢ,pᵢ,d}}
    region = 1:d
    X      = Array{Float64,d}(undef, nᵢ...) 
    Y      = Array{Complex{Float64},d}(undef, FFTW.rfft_output_size(X, region)...)
    mlt    = fft_mult(F)

    FT            = plan_rfft(X, region; flags=FFTW.ESTIMATE) 
    normalize_FT   = one(eltype(X)) * mlt

    IFT           = plan_brfft(FT*X, nᵢ[1], region; flags=FFTW.ESTIMATE) 
    normalize_IFT  = FFTW.normalization(X, region) / mlt

    return rFFTholder{Float64,d}(FT,IFT,normalize_FT,normalize_IFT)
end

@generated function plan(::Type{F}) where {nᵢ, pᵢ, d, F<:cFourierTransform{nᵢ,pᵢ,d}}
    region = 1:d
    X      = Array{Complex{Float64},d}(undef, nᵢ...) 
    Y      = Array{Complex{Float64},d}(undef, nᵢ...) 
    mlt    = fft_mult(F)

    FT            = plan_fft(X, region; flags=FFTW.ESTIMATE) 
    normalize_FT   = one(eltype(X)) * mlt

    IFT           = plan_bfft(FT*X, region; flags=FFTW.ESTIMATE) 
    normalize_IFT  = FFTW.normalization(X, region) / mlt

    return cFFTholder{Float64,d}(FT,IFT,normalize_FT,normalize_IFT)
end

# broadcasting over last dimensions

@generated function plan(::Type{F}, ::Type{LastDimSize{t}}) where {nt, t<:NTuple{nt,Int}, nᵢ, pᵢ, d, F<:rFourierTransform{nᵢ,pᵢ,d}}
    region = 1:d
    nᵢt    = tuple(nᵢ... ,t...)
    X      = Array{Float64,d+nt}(undef, nᵢt...) 
    Y      = Array{Complex{Float64},d+nt}(undef, FFTW.rfft_output_size(X, region)...)
    mlt    = fft_mult(F)

    FT            = plan_rfft(X, region; flags=FFTW.ESTIMATE) 
    normalize_FT   = one(eltype(X)) * mlt

    IFT           = plan_brfft(FT*X, nᵢ[1], region; flags=FFTW.ESTIMATE) 
    normalize_IFT  = FFTW.normalization(X, region) / mlt

    return rFFTholder{Float64,d+nt}(FT,IFT,normalize_FT,normalize_IFT)
end



@generated function plan(::Type{F}, ::Type{LastDimSize{t}}) where {nt, t<:NTuple{nt,Int}, nᵢ, pᵢ, d, F<:cFourierTransform{nᵢ,pᵢ,d}}
    region = 1:d
    nᵢt    = tuple(nᵢ... ,t...)
    X      = Array{Complex{Float64},d+nt}(undef, nᵢt...) 
    Y      = Array{Complex{Float64},d+nt}(undef, nᵢt...) 
    mlt    = fft_mult(F)

    FT             = plan_fft(X, region; flags=FFTW.ESTIMATE) 
    normalize_FT   = one(eltype(X)) * mlt

    IFT            = plan_bfft(FT*X, region; flags=FFTW.ESTIMATE) 
    normalize_IFT  = FFTW.normalization(X, region) / mlt

    return cFFTholder{Float64,d+nt}(FT,IFT,normalize_FT,normalize_IFT)
end


# switching from real plan <-> complex plan
# note: plan(rFourierTransform{nᵢ,pᵢ,d}) defaults to mlt == 1
# note: plan(cFourierTransform{nᵢ,pᵢ,d}) defaults to mlt == 1


@generated function rplan(::Type{F}) where {nᵢ, pᵢ, d, F<:FourierTransform{nᵢ,pᵢ,d}} 
    p   = plan(rFourierTransform{nᵢ,pᵢ,d})
    mlt = fft_mult(F)
    return rFFTholder{Float64,d}(p.FT,p.IFT,p.normalize_FT * mlt, p.normalize_IFT / mlt)
end
@generated function rplan(::Type{F}, ::Type{LastDimSize{t}}) where {nt, t<:NTuple{nt,Int}, nᵢ, pᵢ, d, F<:FourierTransform{nᵢ,pᵢ,d}} 
    p   = plan(rFourierTransform{nᵢ,pᵢ,d}, LastDimSize{t})
    mlt = fft_mult(F)
    return rFFTholder{Float64,d+nt}(p.FT,p.IFT,p.normalize_FT * mlt, p.normalize_IFT / mlt)
end


@generated function cplan(::Type{F}) where {nᵢ, pᵢ, d, F<:FourierTransform{nᵢ,pᵢ,d}} 
    p   = plan(cFourierTransform{nᵢ,pᵢ,d})
    mlt = fft_mult(F)
    return cFFTholder{Float64,d}(p.FT,p.IFT,p.normalize_FT * mlt, p.normalize_IFT / mlt)
end
@generated function cplan(::Type{F}, ::Type{LastDimSize{t}}) where {nt, t<:NTuple{nt,Int}, nᵢ, pᵢ, d, F<:FourierTransform{nᵢ,pᵢ,d}} 
    p   = plan(cFourierTransform{nᵢ,pᵢ,d}, LastDimSize{t})
    mlt = fft_mult(F)
    return cFFTholder{Float64,d+nt}(p.FT,p.IFT,p.normalize_FT * mlt, p.normalize_IFT / mlt)
end



#%% fallback default fft_mult used in the plan
#%% -------------------------------------------------------------

fft_mult(::Type{F}) where F<:FourierTransform = 1

#%% Grid struct ... container for grid information of F<:FourierTransform{nᵢ,pᵢ,d}
#%% ============================================================

struct Grid{nᵢ,pᵢ,dim}
    Δxi::NTuple{dim,Float64}
    Δki::NTuple{dim,Float64}
    xi::NTuple{dim,Vector{Float64}}
    ki::NTuple{dim,Vector{Float64}}
    nyqi::NTuple{dim,Float64}
    Ωx::Float64
    Ωk::Float64
    # the following are redundant but convenient for quick access to nᵢ,pᵢ,dim
    nki::NTuple{dim,Int} # == tuple of dimensions for the rFFT
    nxi::NTuple{dim,Int} # == nᵢ
    periodi::NTuple{dim,Float64} # == pᵢ
    d::Int # == dim
end

function wavenumber(::Type{F}) where F<:FourierTransform
    g = Grid(F)
    λ = zeros(Float64, g.nki)
    for I ∈ CartesianIndices(λ)
        λ[I] = sqrt(sum(abs2, getindex.(g.ki,I.I)))
    end
    λ
end

function frequencies(::Type{F}, i::Int) where F<:FourierTransform
    g = Grid(F)
    kifull = zeros(Float64, g.nki)
    for I ∈ CartesianIndices(kifull)
        kifull[I] = getindex(g.ki[i], I.I[i])
    end
    kifull
end

frequencies(::Type{F}) where {nᵢ, pᵢ, d, F<:FourierTransform{nᵢ,pᵢ,d}} = map(i->frequencies(F,i), tuple(1:d...))::NTuple{d,Array{Float64,d}}

function pixels(::Type{F}, i::Int) where F<:FourierTransform
    g = Grid(F)
    xifull = zeros(Float64, g.nxi)
    for I ∈ CartesianIndices(xifull)
        xifull[I] = getindex(g.xi[i], I.I[i])
    end
    xifull
end

pixels(::Type{F}) where {nᵢ, pᵢ, d, F<:FourierTransform{nᵢ,pᵢ,d}} = map(i->pixels(F,i), tuple(1:d...))::NTuple{d,Array{Float64,d}}


#%% util
#%% ============================================================

"""
` nᵢ, pᵢ, d = _get_npd(;nᵢ, pᵢ=nothing, Δxᵢ=nothing)` is used primarily to to check dimensions are valid
"""
function _get_npd(;nᵢ, pᵢ=nothing, Δxᵢ=nothing)
    @assert !(isnothing(pᵢ) & isnothing(Δxᵢ)) "either pᵢ or Δxᵢ needs to be specified (note: pᵢ = Δxᵢ .* nᵢ)"
    d = length(nᵢ)
    if isnothing(pᵢ)
        @assert d == length(Δxᵢ) "Δxᵢ and nᵢ need to be tuples of the same length"
        pᵢ = tuple((prod(xn) for xn in zip(Δxᵢ,nᵢ))...)
    end
    @assert d == length(pᵢ) "pᵢ and nᵢ need to be tuples of the same length"
    nᵢ, pᵢ, d
end

"""
`k_pre = _fft_output_index_2_freq.(1:n, n, p)` computes the 1-d frequencies
"""
function _fft_output_index_2_freq(ind, nside, period)
    kpre = (2π / period) * (ind - 1)  
    nyq  = (2π / period) * (nside/2)

    # • Both of the following options are valid
    # • Both options return the same value when nside is odd
    # • Using (kpre <= nyq) sets the convention  that the 
    #   frequencies fall in (-nyq, nyq] when nside is even
    # • Using (kpre < nyq) sets the convention  that the 
    #   frequencies fall in [-nyq, nyq) when nside is even
    #----------------
    # • Here are the two options: 
    return  ifelse(kpre <= nyq, kpre, kpre - 2nyq) # option 1
    # return ifelse(kpre < nyq, kpre, kpre - 2nyq)  # option 2
end
