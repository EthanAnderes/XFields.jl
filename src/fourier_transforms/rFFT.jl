#%% rFFT and rFFTunitary are concrete subtypes of c2cTransforms
#%% ============================================================

#%% constructors
function rFFT(::Type{T}=Float64; nᵢ, pᵢ=nothing, Δxᵢ=nothing) where {T<:Real}
    nᵢ,pᵢ′,d = _get_npd(;nᵢ=nᵢ, pᵢ=pᵢ, Δxᵢ=Δxᵢ)
    pᵢ′′ = T.(pᵢ′)
    rFFT{T,nᵢ,pᵢ′′,d}
end
function rFFTunitary(::Type{T}=Float64; nᵢ, pᵢ=nothing, Δxᵢ=nothing) where {T<:Real}
    nᵢ,pᵢ′,d = _get_npd(;nᵢ=nᵢ, pᵢ=pᵢ, Δxᵢ=Δxᵢ)
    pᵢ′′ = T.(pᵢ′)
    rFFTunitary{T,nᵢ,pᵢ′′,d}
end


#%% basic functionality
(*)(::Type{F}, x::Array) where F<:rFFTgeneric = plan(F) * x
(\)(::Type{F}, x::Array) where F<:rFFTgeneric = plan(F) \ x

#%% used in fourier_transforms/plan's
function fft_mult(::Type{F}) where {T,nᵢ,pᵢ,dnᵢ,F<:rFFT{T,nᵢ,pᵢ,dnᵢ}} 
    T(prod(Δx / √(2π) for Δx ∈ Grid(F).Δxi)) 
end
function fft_mult(::Type{F}) where {T,nᵢ,pᵢ,dnᵢ,F<:rFFTunitary{T,nᵢ,pᵢ,dnᵢ}} 
    T(prod(1 / √(n) for n ∈ nᵢ)) 
end

#%% specify the corresponding grid geometry
@generated function Grid(::Type{<:rFFTgeneric{T,nᵢ,pᵢ,dnᵢ}}) where {T,nᵢ,pᵢ,dnᵢ}
    y = map(nᵢ, pᵢ, 1:dnᵢ) do n, p, i
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
    return Grid{T,nᵢ,pᵢ,dnᵢ}(Δxi, Δki, xi, ki, nyqi, Ωx, Ωk, nki, nxi, pᵢ, dnᵢ)
end
 
#%% Used for constructing the covariance matrix of a subset of frequencies
function get_rFFTimpulses(::Type{F}) where {T,nᵢ,pᵢ,dnᵢ,F<:rFFTgeneric{T,nᵢ,pᵢ,dnᵢ}}
    g  = Grid(F)
    CI = CartesianIndices(Base.OneTo.(g.nki))
    LI = LinearIndices(Base.OneTo.(g.nki))

    function _get_dual_k(k,n) 
        dk = n-k+2
        mod1(dk,n)
    end 

    function get_dual_ci(ci::CartesianIndex{dnᵢ}) 
        return CartesianIndex(map(_get_dual_k, ci.I, g.nxi))
    end 

    function rFFTimpulses(ci::CartesianIndex{dnᵢ})
        rimpls = zeros(Complex{T}, g.nki...)
        cimpls = zeros(Complex{T}, g.nki...)
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

