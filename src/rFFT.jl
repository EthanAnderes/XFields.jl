
## ============== rFFT and rFFTunitary ====================
struct rFFT{nᵢ,pᵢ,d}        <: rFourierTransform{nᵢ,pᵢ,d}  end
struct rFFTunitary{nᵢ,pᵢ,d} <: rFourierTransform{nᵢ,pᵢ,d}  end


function rFFT(;nᵢ, pᵢ=nothing, Δxᵢ=nothing) 
    nᵢ,pᵢ,d = _get_npd(;nᵢ=nᵢ, pᵢ=pᵢ, Δxᵢ=Δxᵢ)
    rFFT{nᵢ,pᵢ,d}
end
function rFFTunitary(;nᵢ, pᵢ=nothing, Δxᵢ=nothing) 
    nᵢ,pᵢ,d = _get_npd(;nᵢ=nᵢ, pᵢ=pᵢ, Δxᵢ=Δxᵢ)
    rFFTunitary{nᵢ,pᵢ,d}
end

fft_mult(::Type{F}) where F<:rFFT{nᵢ,pᵢ,d}        where {nᵢ,pᵢ,d} = prod(Δx / √(2π) for Δx ∈ Grid(F).Δxi) 
fft_mult(::Type{F}) where F<:rFFTunitary{nᵢ,pᵢ,d} where {nᵢ,pᵢ,d} = prod(1 / √(n) for n ∈ nᵢ) 

@generated function Grid(::Type{F}) where F<:Union{rFFT{nᵢ,pᵢ,d},rFFTunitary{nᵢ,pᵢ,d}} where {nᵢ,pᵢ,d}
    y = map(nᵢ, pᵢ, 1:d) do n, p, i
        Δx     = p/n
        Δk     = 2π/p
        nyq    = 2π/(2Δx)
        x      = (0:n-1) * Δx
        k_pre = _fft_output_index_2_freq.(1:n, n, p)
        k      = (i == 1) ? k_pre[1:(n÷2+1)] : k_pre
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
 
