#%% Transform types
#%% ============================================================

abstract type Transform end

# nᵢ ≡ number of grid evaluations to a side
# pᵢ ≡ period (e.g. grid evals at {0, pᵢ/nᵢ, pᵢ2/nᵢ, …, pᵢ(nᵢ-1)/nᵢ})
abstract type FourierTransform{T,nᵢ,pᵢ,dnᵢ}  <: Transform end
abstract type rFourierTransform{T<:Real,nᵢ,pᵢ,dnᵢ} <: FourierTransform{T,nᵢ,pᵢ,dnᵢ} end
abstract type cFourierTransform{T<:Real,nᵢ,pᵢ,dnᵢ} <: FourierTransform{T,nᵢ,pᵢ,dnᵢ} end

# Note: using these aliases to allow reveral of order dnᵢ <-> nᵢ
rFT{T,dnᵢ,nᵢ} = rFourierTransform{T,nᵢ,pᵢ,dnᵢ} where {pᵢ} 
cFT{T,dnᵢ,nᵢ} = cFourierTransform{T,nᵢ,pᵢ,dnᵢ} where {pᵢ} 
FT{T,dnᵢ,nᵢ}  = FourierTransform{T,nᵢ,pᵢ,dnᵢ} where {pᵢ}


#  This allows broadcasting an fft plan to slices indexed by trailing dimensions
#  note: tᵢ <: NTuple{dtᵢ,Int}
abstract type LastDimSize{tᵢ,dtᵢ} end
LDimS{dtᵢ} = LastDimSize{tᵢ,dtᵢ} where {tᵢ}



#%% Low level container for forward/backward plans and normalization
#%% -------------------------------------------------------------
#%% Instances of these containers know how to mult and divide


# Here we 
struct rFFTholder{T<:Real,dnᵢdtᵢ}
    FT::FFTW.rFFTWPlan{T,-1,false,dnᵢdtᵢ}
    IFT::FFTW.rFFTWPlan{Complex{T},1,false,dnᵢdtᵢ}
    normalize_FT::T
    normalize_IFT::T
end
struct Adjoint_rFFTholder{T<:Real,dnᵢdtᵢ}
    FT::FFTW.rFFTWPlan{T,-1,false,dnᵢdtᵢ}
    IFT::FFTW.rFFTWPlan{Complex{T},1,false,dnᵢdtᵢ}
    normalize_FT::T
    normalize_IFT::T
end


struct cFFTholder{T<:Real,dnᵢdtᵢ} # dnᵢdtᵢ is the total dimension of the array it operates on
    FT::FFTW.cFFTWPlan{Complex{T},-1,false,dnᵢdtᵢ}
    IFT::FFTW.cFFTWPlan{Complex{T},1,false,dnᵢdtᵢ}
    normalize_FT::T
    normalize_IFT::T
end
struct Adjoint_cFFTholder{T<:Real,dnᵢdtᵢ} # dnᵢdtᵢ is the total dimension of the array it operates on
    FT::FFTW.cFFTWPlan{Complex{T},-1,false,dnᵢdtᵢ}
    IFT::FFTW.cFFTWPlan{Complex{T},1,false,dnᵢdtᵢ}
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



adjoint(p::rFFTholder{T,dnᵢdtᵢ}) where {T,dnᵢdtᵢ} = Adjoint_rFFTholder{T,dnᵢdtᵢ}(p.FT,p.IFT,p.normalize_FT,p.normalize_IFT)
adjoint(p::cFFTholder{T,dnᵢdtᵢ}) where {T,dnᵢdtᵢ} = Adjoint_cFFTholder{T,dnᵢdtᵢ}(p.FT,p.IFT,p.normalize_FT,p.normalize_IFT)

transpose(p::rFFTholder) = p
transpose(p::cFFTholder) = p


#%% Plans (TODO: get rid of the allocation for the planned ffts)
#%% -------------------------------------------------------------
 

@generated function rplan(::Type{F}, ::Type{L}) where {T,nᵢ,dnᵢ,tᵢ,dtᵢ,F<:FT{T,dnᵢ,nᵢ},L<:LastDimSize{tᵢ,dtᵢ}}
    region = 1:dnᵢ
    nᵢtᵢ    = tuple(nᵢ... ,tᵢ...)
    dnᵢdtᵢ = dnᵢ+dtᵢ
    X      = Array{T,dnᵢdtᵢ}(undef, nᵢtᵢ...) 
    Y      = Array{Complex{T},dnᵢdtᵢ}(undef, FFTW.rfft_output_size(X, region)...)
    mlt    = fft_mult(F)

    FT            = plan_rfft(X, region; flags=FFTW.ESTIMATE) 
    normalize_FT  = one(eltype(X)) * mlt
    IFT           = plan_brfft(FT*X, nᵢ[1], region; flags=FFTW.ESTIMATE) 
    normalize_IFT = FFTW.normalization(X, region) / mlt

    return rFFTholder{T,dnᵢdtᵢ}(FT,IFT,normalize_FT,normalize_IFT)
end

function rplan(::Type{F}) where {F<:FT}
    rplan(F, LastDimSize{(),0})
end


@generated function cplan(::Type{F}, ::Type{L}) where {T,nᵢ,dnᵢ,tᵢ,dtᵢ,F<:FT{T,dnᵢ,nᵢ},L<:LastDimSize{tᵢ,dtᵢ}}
    region = 1:dnᵢ
    nᵢtᵢ    = tuple(nᵢ... ,tᵢ...)
    dnᵢdtᵢ = dnᵢ+dtᵢ
    X      = Array{Complex{T},dnᵢdtᵢ}(undef, nᵢtᵢ...) 
    Y      = Array{Complex{T},dnᵢdtᵢ}(undef, nᵢtᵢ...) 
    mlt    = fft_mult(F)

    FT            = plan_fft(X, region; flags=FFTW.ESTIMATE) 
    normalize_FT  = one(eltype(X)) * mlt
    IFT           = plan_bfft(FT*X, region; flags=FFTW.ESTIMATE) 
    normalize_IFT = FFTW.normalization(X, region) / mlt

    return cFFTholder{T,dnᵢdtᵢ}(FT,IFT,normalize_FT,normalize_IFT)
end

function cplan(::Type{F}) where {F<:FT}
    cplan(F, LastDimSize{(),0})
end


plan(::Type{F}, ::Type{L}) where {F<:rFT, L<:LastDimSize} = rplan(F,L)
plan(::Type{F}, ::Type{L}) where {F<:cFT, L<:LastDimSize} = cplan(F,L)

plan(::Type{F}) where {F<:rFT} = rplan(F)
plan(::Type{F}) where {F<:cFT} = cplan(F)



#%% fallback default fft_mult used in the plan
#%% -------------------------------------------------------------

fft_mult(::Type{F}) where F<:FourierTransform = 1

#%% Grid struct ... container for grid information of F<:FourierTransform{nᵢ,pᵢ,d}
#%% ============================================================

struct Grid{T,nᵢ,pᵢ,dnᵢ}
    Δxi::NTuple{dnᵢ,T}
    Δki::NTuple{dnᵢ,T}
    xi::NTuple{dnᵢ,Vector{T}}
    ki::NTuple{dnᵢ,Vector{T}}
    nyqi::NTuple{dnᵢ,T}
    Ωx::T
    Ωk::T
    # the following are redundant but convenient for quick access to nᵢ,pᵢ,dnᵢ
    nki::NTuple{dnᵢ,Int} # == tuple of dnᵢensions for the rFFT
    nxi::NTuple{dnᵢ,Int} # == nᵢ
    periodi::NTuple{dnᵢ,T} # == pᵢ
    d::Int # == dnᵢ
end

function wavenumber(::Type{F}) where {T,d,F<:FT{T,d}}
    g = Grid(F)
    λ = zeros(T, g.nki)
    for I ∈ CartesianIndices(λ)
        λ[I] = sqrt(sum(abs2, getindex.(g.ki,I.I)))
    end
    λ
end

function frequencies(::Type{F}, i::Int) where {T,d,F<:FT{T,d}}
    g = Grid(F)
    kifull = zeros(T, g.nki)
    for I ∈ CartesianIndices(kifull)
        kifull[I] = getindex(g.ki[i], I.I[i])
    end
    kifull
end

function frequencies(::Type{F}) where {T,d,F<:FT{T,d}}
    g = Grid(F) 
    map(i->frequencies(F,i), tuple(1:d...))::NTuple{d,Array{T,d}}
end

function pixels(::Type{F}, i::Int) where {T,d,F<:FT{T,d}}
    g = Grid(F)
    xifull = zeros(T, g.nxi)
    for I ∈ CartesianIndices(xifull)
        xifull[I] = getindex(g.xi[i], I.I[i])
    end
    xifull
end

function pixels(::Type{F})  where {T,d,F<:FT{T,d}}
    g = Grid(F) 
    map(i->pixels(F,i), tuple(1:d...))::NTuple{d,Array{T,d}}
end

#%% util
#%% ============================================================
#TODO: upgrade so it works for general T 

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
