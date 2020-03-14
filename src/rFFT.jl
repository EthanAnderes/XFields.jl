#%% rFFT and rFFTunitary for real fft operations
#%% ============================================================

struct rFFT{nᵢ,pᵢ,d}        <: rFourierTransform{nᵢ,pᵢ,d}  end
struct rFFTunitary{nᵢ,pᵢ,d} <: rFourierTransform{nᵢ,pᵢ,d}  end

#%% basic functionality
(*)(::Type{FT}, x::Array) where FT<:Union{rFFT, rFFTunitary} = plan(FT) * x
(\)(::Type{FT}, x::Array) where FT<:Union{rFFT, rFFTunitary} = plan(FT) \ x

#%% constructors
function rFFT(;nᵢ, pᵢ=nothing, Δxᵢ=nothing) 
    nᵢ,pᵢ,d = _get_npd(;nᵢ=nᵢ, pᵢ=pᵢ, Δxᵢ=Δxᵢ)
    rFFT{nᵢ,pᵢ,d}
end
function rFFTunitary(;nᵢ, pᵢ=nothing, Δxᵢ=nothing) 
    nᵢ,pᵢ,d = _get_npd(;nᵢ=nᵢ, pᵢ=pᵢ, Δxᵢ=Δxᵢ)
    rFFTunitary{nᵢ,pᵢ,d}
end

#%% used in fourier_transforms/plan's
fft_mult(::Type{F}) where F<:rFFT{nᵢ,pᵢ,d}        where {nᵢ,pᵢ,d} = prod(Δx / √(2π) for Δx ∈ Grid(F).Δxi) 
fft_mult(::Type{F}) where F<:rFFTunitary{nᵢ,pᵢ,d} where {nᵢ,pᵢ,d} = prod(1 / √(n) for n ∈ nᵢ) 

#%% specify the corresponding grid geometry
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
 
#%% Used for constructing the covariance matrix of a subset of frequencies
function get_rFFTimpulses(::Type{rFT}) where {nᵢ,pᵢ,dim,rFT<:rFourierTransform{nᵢ,pᵢ,dim}} 
    rg = Grid(rFT)
    CI = CartesianIndices(Base.OneTo.(rg.nki))
    LI = LinearIndices(Base.OneTo.(rg.nki))

    function _get_dual_k(k,n) 
        dk = n-k+2
        mod1(dk,n)
    end 

    function get_dual_ci(ci::CartesianIndex{dim}) 
        return CartesianIndex(map(_get_dual_k, ci.I, rg.nxi))
    end 

    function rFFTimpulses(ci::CartesianIndex{dim})
        rimpls = zeros(Complex{Float64}, rg.nki...)
        cimpls = zeros(Complex{Float64}, rg.nki...)
        dual_ci = get_dual_ci(ci)
        if (ci==first(CI)) || (ci==dual_ci)
            rimpls[ci]  = 1
        elseif dual_ci ∈ CI
            rimpls[ci]  = 1/2
            cimpls[ci]  = im/2
            rimpls[dual_ci]  =  1/2
            cimpls[dual_ci]  = -im/2
        else
            rimpls[ci]  = 1/2
            cimpls[ci]  = im/2
        end
        return rimpls, cimpls
    end

    return rFFTimpulses, CI, LI, get_dual_ci
end

