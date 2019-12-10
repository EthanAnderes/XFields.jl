## =====================================================
abstract type Transform end

## =====================================================
# nᵢ ≡ number of grid evaluations to a side
# pᵢ ≡ period (e.g. grid evals at {0, pᵢ/nᵢ, pᵢ2/nᵢ, …, pᵢ(nᵢ-1)/nᵢ})

abstract type FourierTransform{nᵢ,pᵢ,d}  <: Transform end
abstract type rFourierTransform{nᵢ,pᵢ,d} <: FourierTransform{nᵢ,pᵢ,d} end
abstract type cFourierTransform{nᵢ,pᵢ,d} <: FourierTransform{nᵢ,pᵢ,d} end

## =====================================================
#  This allows broadcasting an fft plan to slices indexed by trailing dimensions
#  note: t isa Int or a tuple of ints
abstract type LastDimSize{t} end


## =====================================================
# generic multiply
(*)(::Type{FT}, x::Array) where FT<:FourierTransform = plan(FT) * x
(\)(::Type{FT}, x::Array) where FT<:FourierTransform = plan(FT) \ x


## =====================================================
#  Generated functions for the plans

@generated function plan(::Type{F}) where {nᵢ, pᵢ, d, F<:rFourierTransform{nᵢ,pᵢ,d}}
    FFT  =  fft_mult(F) * plan_rfft(Array{Float64,d}(undef, nᵢ...); flags=FFTW.ESTIMATE)
    return :( $FFT )
end

@generated function plan(::Type{F}) where {nᵢ, pᵢ, d, F<:cFourierTransform{nᵢ,pᵢ,d}}
    FFT  =  fft_mult(F) * plan_fft(Array{Complex{Float64},d}(undef, nᵢ...); flags=FFTW.ESTIMATE)
    return :( $FFT )
end

@generated function cplan(::Type{F}) where {nᵢ, pᵢ, d, F<:FourierTransform{nᵢ,pᵢ,d}} 
    FFT  =  fft_mult(F) * plan(cFourierTransform{nᵢ,pᵢ,d})
    return :( $FFT )
end

@generated function rplan(::Type{F}) where {nᵢ, pᵢ, d, F<:FourierTransform{nᵢ,pᵢ,d}} 
    FFT  =  fft_mult(F) * plan(rFourierTransform{nᵢ,pᵢ,d})
    return :( $FFT )
end


@generated function plan(::Type{F}, ::Type{LastDimSize{t}}) where {t, nᵢ, pᵢ, d, F<:rFourierTransform{nᵢ,pᵢ,d}}
    FFT  =  fft_mult(F) * plan_rfft(Array{Float64,d+length(t)}(undef, nᵢ... ,t...), 1:d; flags=FFTW.ESTIMATE)
    return :( $FFT )
end

@generated function plan(::Type{F}, ::Type{LastDimSize{t}}) where {t, nᵢ, pᵢ, d, F<:cFourierTransform{nᵢ,pᵢ,d}}
    FFT  =  fft_mult(F) * plan_fft(Array{Complex{Float64},d+length(t)}(undef, nᵢ..., t...), 1:d; flags=FFTW.ESTIMATE)
    return :( $FFT )
end

@generated function cplan(::Type{F}, ::Type{LastDimSize{t}}) where {t, nᵢ, pᵢ, d, F<:FourierTransform{nᵢ,pᵢ,d}} 
    FFT  =  fft_mult(F) * plan(cFourierTransform{nᵢ,pᵢ,d}, LastDimSize{t})
    return :( $FFT )
end

@generated function rplan(::Type{F}, ::Type{LastDimSize{t}}) where {t, nᵢ, pᵢ, d, F<:FourierTransform{nᵢ,pᵢ,d}} 
    FFT  =  fft_mult(F) * plan(rFourierTransform{nᵢ,pᵢ,d}, LastDimSize{t})
    return :( $FFT )
end



# fallback default
fft_mult(::Type{F}) where F<:FourierTransform = 1





## =====================================================
# container for grid information of F<:FourierTransform{nᵢ,pᵢ,d}

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


## =====================================================
# misc ...

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

function adjoint(F::FFTW.ScaledPlan)
    iF = inv(F)
    return (F.scale / iF.scale) * iF
end

transpose(F::FFTW.ScaledPlan) = F
