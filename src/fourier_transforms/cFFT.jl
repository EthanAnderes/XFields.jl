#%% cFFT and cFFTunitary are concrete subtypes of c2cTransforms
#%% ============================================================

struct cFFT{T<:Real,nᵢ,pᵢ,dnᵢ}        <: c2cFourierTransform{T,dnᵢ,nᵢ}  end
struct cFFTunitary{T<:Real,nᵢ,pᵢ,dnᵢ} <: c2cFourierTransform{T,dnᵢ,nᵢ}  end

#%% constructors
function cFFT(::Type{T}=Float64; nᵢ, pᵢ=nothing, Δxᵢ=nothing) where {T<:Real}
    nᵢ,pᵢ′,d = _get_npd(;nᵢ=nᵢ, pᵢ=pᵢ, Δxᵢ=Δxᵢ)
    pᵢ′′ = T.(pᵢ′)
    cFFT{T,nᵢ,pᵢ′′,d}
end
function cFFTunitary(::Type{T}=Float64; nᵢ, pᵢ=nothing, Δxᵢ=nothing) where {T<:Real}
    nᵢ,pᵢ′,d = _get_npd(;nᵢ=nᵢ, pᵢ=pᵢ, Δxᵢ=Δxᵢ)
    pᵢ′′ = T.(pᵢ′)
    cFFTunitary{T,nᵢ,pᵢ′′,d}
end


#%% basic functionality
(*)(::Type{F}, x::Array) where F<:Union{cFFT, cFFTunitary} = plan(F) * x
(\)(::Type{F}, x::Array) where F<:Union{cFFT, cFFTunitary} = plan(F) \ x

#%% used in fourier_transforms/plan's
function fft_mult(::Type{F}) where {T<:Real,nᵢ,pᵢ,dnᵢ,F<:cFFT{T,nᵢ,pᵢ,dnᵢ}}  
    T(prod(Δx / √(2π) for Δx ∈ Grid(F).Δxi)) 
end
function fft_mult(::Type{F}) where {T<:Real,nᵢ,pᵢ,dnᵢ,F<:cFFTunitary{T,nᵢ,pᵢ,dnᵢ}} 
    T(prod(1 / √(n) for n ∈ nᵢ)) 
end 

#%% specify the corresponding grid geometry
@generated function Grid(::Type{F}) where {T<:Real,nᵢ,pᵢ,dnᵢ,F<:Union{cFFT{T,nᵢ,pᵢ,dnᵢ},cFFTunitary{T,nᵢ,pᵢ,dnᵢ}}}
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
    return Grid{T,nᵢ,pᵢ,dnᵢ}(Δxi, Δki, xi, ki, nyqi, Ωx, Ωk, nki, nxi, pᵢ, dnᵢ)
end
 

