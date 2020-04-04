
#%% abstract Fourier transform types which contain enough information
#%% to allow plans to be generated (modulo the constant multiplier)
#%% ==================================================================
abstract type r2cFourierTransform{T<:Real,dnᵢ,nᵢ}  <: r2cTransform{T,dnᵢ,nᵢ} end
abstract type c2cFourierTransform{T<:Real,dnᵢ,nᵢ}  <: c2cTransform{T,dnᵢ,nᵢ} end
FourierTransform{T,dnᵢ,nᵢ} = Union{r2cFourierTransform{T,dnᵢ,nᵢ},c2cFourierTransform{T,dnᵢ,nᵢ}}


#%% fallback default fft_mult used in the plan
#%% ==================================================================
fft_mult(::Type{F}) where F<:FourierTransform = 1


#%% Low level container for forward/backward plans and normalization
#%% ==================================================================
#%% Instances of these containers know how to mult and divide

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


#%% Plans constructed from FourierTransform types
#%% ==================================================================

abstract type LastDimSize{tᵢ,dtᵢ} end

@generated function rplan(::Type{F}, ::Type{L}) where {T<:Real,dnᵢ,dtᵢ,nᵢ,tᵢ,F<:FourierTransform{T,dnᵢ,nᵢ},L<:LastDimSize{tᵢ,dtᵢ}}
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

@generated function cplan(::Type{F}, ::Type{L}) where {T<:Real,dnᵢ,dtᵢ,nᵢ,tᵢ,F<:FourierTransform{T,dnᵢ,nᵢ},L<:LastDimSize{tᵢ,dtᵢ}}
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

function rplan(::Type{F}) where {T<:Real,dnᵢ,nᵢ,F<:FourierTransform{T,dnᵢ,nᵢ}}
    rplan(F, LastDimSize{(),0})
end

function cplan(::Type{F}) where {T<:Real,dnᵢ,nᵢ,F<:FourierTransform{T,dnᵢ,nᵢ}}
    cplan(F, LastDimSize{(),0})
end

plan(::Type{F}, ::Type{L}) where {F<:r2cFourierTransform, L<:LastDimSize} = rplan(F,L)
plan(::Type{F}, ::Type{L}) where {F<:c2cFourierTransform, L<:LastDimSize} = cplan(F,L)

plan(::Type{F}) where {F<:r2cFourierTransform} = rplan(F)
plan(::Type{F}) where {F<:c2cFourierTransform} = cplan(F)


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

function wavenumber(::Type{F}) where {T,d,F<:FourierTransform{T,d}}
    g = Grid(F)
    λ = zeros(T, g.nki)
    for I ∈ CartesianIndices(λ)
        λ[I] = sqrt(sum(abs2, getindex.(g.ki,I.I)))
    end
    λ
end

function frequencies(::Type{F}, i::Int) where {T,d,F<:FourierTransform{T,d}}
    g = Grid(F)
    kifull = zeros(T, g.nki)
    for I ∈ CartesianIndices(kifull)
        kifull[I] = getindex(g.ki[i], I.I[i])
    end
    kifull
end

function frequencies(::Type{F}) where {T,d,F<:FourierTransform{T,d}}
    g = Grid(F) 
    map(i->frequencies(F,i), tuple(1:d...))::NTuple{d,Array{T,d}}
end

function pixels(::Type{F}, i::Int) where {T,d,F<:FourierTransform{T,d}}
    g = Grid(F)
    xifull = zeros(T, g.nxi)
    for I ∈ CartesianIndices(xifull)
        xifull[I] = getindex(g.xi[i], I.I[i])
    end
    xifull
end

function pixels(::Type{F})  where {T,d,F<:FourierTransform{T,d}}
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
#
