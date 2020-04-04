#%% cFFT and cFFTunitary for full fft operations
#%% ============================================================

struct cFFT{nᵢ,pᵢ,dnᵢ}        <: cFourierTransform{Float64,nᵢ,pᵢ,dnᵢ}  end
struct cFFTunitary{nᵢ,pᵢ,dnᵢ} <: cFourierTransform{Float64,nᵢ,pᵢ,dnᵢ}  end

#%% constructors
function cFFT(;nᵢ, pᵢ=nothing, Δxᵢ=nothing) 
    nᵢ,pᵢ,d = _get_npd(;nᵢ=nᵢ, pᵢ=pᵢ, Δxᵢ=Δxᵢ)
    cFFT{nᵢ,pᵢ,d}
end
function cFFTunitary(;nᵢ, pᵢ=nothing, Δxᵢ=nothing) 
    nᵢ,pᵢ,d = _get_npd(;nᵢ=nᵢ, pᵢ=pᵢ, Δxᵢ=Δxᵢ)
    cFFTunitary{nᵢ,p,d}
end


#%% basic functionality
(*)(::Type{F}, x::Array) where F<:Union{cFFT, cFFTunitary} = plan(F) * x
(\)(::Type{F}, x::Array) where F<:Union{cFFT, cFFTunitary} = plan(F) \ x

#%% used in fourier_transforms/plan's
fft_mult(::Type{F}) where {nᵢ,pᵢ,dnᵢ,F<:cFFT{nᵢ,pᵢ,dnᵢ}}        = prod(Δx / √(2π) for Δx ∈ Grid(F).Δxi) 
fft_mult(::Type{F}) where {nᵢ,pᵢ,dnᵢ,F<:cFFTunitary{nᵢ,pᵢ,dnᵢ}} = prod(1 / √(n) for n ∈ nᵢ) 

#%% specify the corresponding grid geometry
@generated function Grid(::Type{F}) where {nᵢ,pᵢ,dnᵢ,F<:Union{cFFT{nᵢ,pᵢ,dnᵢ},cFFTunitary{nᵢ,pᵢ,dnᵢ}}}
    y = map(nᵢ, pᵢ, 1:dnᵢ) do n, p, i
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
    return Grid{Float64,nᵢ,pᵢ,dnᵢ}(Δxi, Δki, xi, ki, nyqi, Ωx, Ωk, nki, nxi, pᵢ, dnᵢ)
end
 

