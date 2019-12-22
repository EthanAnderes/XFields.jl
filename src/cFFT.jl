## =====================================================
# cFFT and cFFTunitary
struct cFFT{nᵢ,pᵢ,d}        <: cFourierTransform{nᵢ,pᵢ,d}  end
struct cFFTunitary{nᵢ,pᵢ,d} <: cFourierTransform{nᵢ,pᵢ,d}  end

Base.:*(::Type{FT}, x::Array) where FT<:Union{cFFT, cFFTunitary} = plan(FT) * x
Base.:\(::Type{FT}, x::Array) where FT<:Union{cFFT, cFFTunitary} = plan(FT) \ x

function cFFT(;nᵢ, pᵢ=nothing, Δxᵢ=nothing) 
    nᵢ,pᵢ,d = _get_npd(;nᵢ=nᵢ, pᵢ=pᵢ, Δxᵢ=Δxᵢ)
    cFFT{nᵢ,pᵢ,d}
end
function cFFTunitary(;nᵢ, pᵢ=nothing, Δxᵢ=nothing) 
    nᵢ,pᵢ,d = _get_npd(;nᵢ=nᵢ, pᵢ=pᵢ, Δxᵢ=Δxᵢ)
    cFFTunitary{nᵢ,pᵢ,d}
end

fft_mult(::Type{F}) where F<:cFFT{nᵢ,pᵢ,d}        where {nᵢ,pᵢ,d} = prod(Δx / √(2π) for Δx ∈ Grid(F).Δxi) 
fft_mult(::Type{F}) where F<:cFFTunitary{nᵢ,pᵢ,d} where {nᵢ,pᵢ,d} = prod(1 / √(n) for n ∈ nᵢ) 

@generated function Grid(::Type{F}) where F<:Union{cFFT{nᵢ,pᵢ,d},cFFTunitary{nᵢ,pᵢ,d}} where {nᵢ,pᵢ,d}
    y = map(nᵢ, pᵢ, 1:d) do n, p, i
        Δx     = p/n
        Δk     = 2π/p
        nyq    = 2π/(2Δx)
        x      = (0:n-1) * Δx
        k      = _fft_output_index_2_freq.(1:n, n, p)
        (Δx=Δx, Δk=Δk, nyq=nyq, x=x, k=k) 
    end
    Δki     = tuple((yi.Δk for yi ∈ y)...)
    Δxi     = tuple((yi.Δx for yi ∈ y)...)
    nyqi    = tuple((yi.nyq for yi ∈ y)...)
    xi      = tuple((yi.x for yi ∈ y)...)
    ki      = tuple((yi.k for yi ∈ y)...) # note: you might need to reverse the order here...
    Ωk      = prod(Δki)
    Ωx      = prod(Δxi)
    nxi     = nᵢ
    nki     = map(length, ki)
    Grid{nᵢ,pᵢ,d}(Δxi, Δki, xi, ki, nyqi, Ωx, Ωk, nki, nxi, pᵢ, d)
end
 

