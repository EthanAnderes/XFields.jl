```julia
using XFields
using FFTransforms
using Spectra

using LinearAlgebra
using PyPlot
```

Define the transform
------------------------------------------


```julia
trn = let
    𝕨 = r𝕎32(256, π) ⊗ 𝕎(256, 4.0)
    ordinary_scale(𝕨)*𝕨
end;
```

Signal and noise spectral density
--------------------------------------------

Here we use Matern spectral densities


```julia
Cf = let trn = trn, ρ = 0.15, ν = 2.1, σ² = 1 / 2
    l  = wavenum(trn)
    d  = ndims(l)
    cl = σ² .* Spectra.matern_spec.(l; rho=ρ, nu=ν, d=d)
    Cf = DiagOp(Xfourier(trn, cl))
    @show pixel_variance = (Cf[:] ./ (2π)^(d/2))[1]
    (σ² / pixel_variance) * Cf
end;
```

    pixel_variance = (Cf[:] ./ (2π) ^ (d / 2))[1] = 0.49999709211062265


Noise spectral density


```julia
Cn =  let trn=trn, μKarcminT=15, ℓknee=4*minimum(Δfreq(trn)), αknee=2

    Ωx_unit = deg2rad(1/60)^2 ## area [rad²] for 1arcmin×1arcmin pixel
    wvn     = wavenum(trn); wvn[1] = Inf
    knee    = @. 1 + XFields.nan2zero((ℓknee / wvn) ^ αknee)
    cnl     = μKarcminT^2 .* Ωx_unit .* knee
    Cn      =  Xfourier(trn, μKarcminT^2 .* Ωx_unit .* knee) |> DiagOp

    Cn
end;
```

Mask and Transfer function linear operators
--------------------------------------------

Mask


```julia
Ma =  let trn=trn, x1bdry = (0.1, 0.9), x2bdry = (0.2,0.95)

    lbr1, rbr1 = trn.period[1] .* x1bdry
    lbr2, rbr2 = trn.period[2] .* x2bdry
    x1, x2 = pix(trn)
    ma = (lbr1 .< x1 .< rbr1) .* (lbr2 .< x2 .< rbr2)'
    Ma = Xmap(trn, ma) |> DiagOp

    Ma
end;
```

Transfer function


```julia
Tr =  let trn=trn, beam_npix = 4

    fwhm_rad = beam_npix * min(Δpix(trn)...)
    beam = l -> exp(-abs2(l * fwhm_rad) / (16*log(2)))
    tr   = beam.(wavenum(trn))
    tr .*= wavenum(trn) .< 0.9nyq(trn)[1]
    Tr   = Xfourier(trn, tr) |> DiagOp

    Tr
end;
```

White noise simulator
------------------------------------


```julia
function ωη(trn::T) where T<:Transform
    zx = randn(eltype_in(trn),size_in(trn))
    Xmap(trn, zx ./ √Ωx(trn))
end
```




    ωη (generic function with 1 method)



Field simulation: signal (`fsim`), noise (`nsim`) and data (`dsim`)
--------------------------------------------------------------


```julia
dsim, fsim, nsim = let trn=trn, Cf=Cf, Cn=Cn, Cf=Cf, Ma=Ma, Tr=Tr

    fsim = √Cf * ωη(trn)
    nsim = √Cn * ωη(trn)

    dsim = Ma * Tr * fsim + Ma * nsim

    # μsim = Xmap(trn,sum(dsim[:]) ./ sum(Ma[:]))
    # dsim = dsim - Ma * μsim

    dsim, fsim, nsim
end;
```

Plots of the signal, noise and data
--------------------------------------------------------------

## Signal `fsim`


```julia
let trn=trn, f=fsim
    fig, ax = subplots(1, figsize=(8,4))
    x1, x2 = pix(trn)
    pcm = ax.pcolormesh(x2, x1, f[:])
    ax.set_title("signal")
    fig.colorbar(pcm, ax = ax)
    fig.tight_layout()
end;
```


![png](example_files/example_19_0.png)


## Noise `nsim`


```julia
let trn=trn, f=nsim
    fig, ax = subplots(1, figsize=(8,4))
    x1, x2 = pix(trn)
    pcm = ax.pcolormesh(x2, x1, f[:])
    ax.set_title("noise")
    fig.colorbar(pcm, ax = ax)
    fig.tight_layout()
end;
```


![png](example_files/example_21_0.png)


## Data `dsim`


```julia
let trn=trn, f=dsim
    fig, ax = subplots(1, figsize=(8,4))
    x1, x2 = pix(trn)
    pcm = ax.pcolormesh(x2, x1, f[:])
    ax.set_title("Data")
    fig.colorbar(pcm, ax = ax)
    fig.tight_layout()
end;
```


![png](example_files/example_23_0.png)


## Mask and transfer


```julia
let trn=trn, Ma=Ma, Tr=Tr
    fig, ax = subplots(2,1, figsize=(8,8))

    x1, x2 = pix(trn)
    pcm1 = ax[1].pcolormesh(x2, x1, Ma[:])
    ax[1].set_title("Pixel mask")

    l1, l2 = freq(trn)
    p2 = sortperm(l2)
    tr = real.(Tr[!])
    pcm2 = ax[2].pcolormesh(l2[p2], l1, tr[:,p2])
    ax[2].set_title("Fourier transfer function")

    fig.colorbar(pcm1, ax=ax[1])
    fig.colorbar(pcm2, ax=ax[2])
    fig.tight_layout()
end;
```


![png](example_files/example_25_0.png)


Bandpowers (i.e. periodogram)
--------------------------------------------------------------


```julia
function power(f::Xfield{F}, g::Xfield{F}; bin::Int=2, kmax=Inf, mult=1) where F<:Transform
    trn     = fieldtransform(f)
    k      = wavenum(trn)
    pwr    = @. mult * real(f[!] * conj(g[!]) + conj(f[!]) * g[!]) / 2
    Δbin   = bin * minimum(Δfreq(trn))
    k_left = 0
    while k_left < min(kmax, maximum(k))
        k_right    = k_left + Δbin
        indx       = findall(k_left .< k .<= k_right)
        pwr[indx] .= sum( pwr[indx] ) / length( pwr[indx] )
        k_left  = k_right
    end
    return pwr
end
```




    power (generic function with 1 method)




```julia
function power(f::Xfield{F}; bin::Int=2, kmax=Inf, mult=1) where F<:Transform
    power(f, f; bin=bin, kmax=kmax, mult=mult)
end
```




    power (generic function with 2 methods)



## Noise and signal bandpowers


```julia
let trn=trn, Cn=Cn, Cf=Cf, f=fsim, n=nsim
    l     = wavenum(trn)
    mult  = l .* (l .+ 1)

    fig, ax = subplots(1, figsize=(8,4))

    pwrf = power(f; mult=mult * Ωk(trn) )
    pwrn = power(n; mult=mult * Ωk(trn) )
    (l[:,1], pwrf[:,1]) |> x->ax.plot(x[1][2:end],x[2][2:end], label="signal")
    (l[:,1], pwrn[:,1]) |> x->ax.plot(x[1][2:end],x[2][2:end], label="noise")

    cf = real.(Cf[!])
    cn = real.(Cn[!])
    (l[:,1], (mult.*cf)[:,1]) |> x->ax.plot(x[1][3:end],x[2][3:end])
    (l[:,1], (mult.*cn)[:,1]) |> x->ax.plot(x[1][3:end],x[2][3:end])

    ax.set_xlabel("wavenumber")
    ax.set_ylabel("power")
    ax.legend()
    fig.tight_layout()
end;
```


![png](example_files/example_30_0.png)


Set up basic  d = M⋅Tf⋅f + M⋅n
--------------------------------------------------------------

custom pcg with function composition (Minv * A \approx I)


```julia
function pcg(Minv::Function, A::Function, b, x=0*b; nsteps::Int=75, rel_tol::Float64 = 1e-8)
    r       = b - A(x)
    z       = Minv(r)
    p       = deepcopy(z)
    res     = dot(r,z)
    reshist = Vector{typeof(res)}()
    for i = 1:nsteps
        Ap        = A(p)
        α         = res / dot(p,Ap)
        x         = x + α * p
        r         = r - α * Ap
        z         = Minv(r)
        res′      = dot(r,z)
        p         = z + (res′ / res) * p
        rel_error = XFields.nan2zero(sqrt(dot(r,r)/dot(b,b)))
        if rel_error < rel_tol
            return x, reshist
        end
        push!(reshist, rel_error)
        res = res′
    end
    return x, reshist
end
```




    pcg (generic function with 2 methods)




```julia
function LinearAlgebra.dot(f::Xfield{FT},g::Xfield{FT}) where FT<:Transform
    trn = fieldtransform(f)
    Ωx(trn) * dot(f[:],g[:])
end
```


```julia
wfsim, wfhist, zwf = let trn=trn, Cn=Cn, Cf=Cf, Tr=Tr, Ma=Ma, dsim=dsim

    A  = Ma * Tr / Cn * Tr * Ma
    B  = 1 / Cf
    P  = inv(Tr / Cn * Tr + B)

    wfsim, wfhist = pcg(
            w -> P * w,
            w -> A * w +  B * w,
            Ma * Tr / Cn * dsim,
            nsteps  = 500,
            rel_tol = 1e-15,
    )

    # compte the chi2 z-score
    dfd   = sum(Ma[:]) # degrees of freedom of the data
    Δ1    = dsim - Ma * wfsim
    zwf  = - dot(Δ1, Cn \ Δ1) - dot(wfsim, Cf \ wfsim)
    zwf -= -dfd
    zwf /= sqrt(2*dfd)


    wfsim, wfhist, zwf

end;
```

the "residual" per iteration


```julia
let wfhist=wfhist
    fig, ax = subplots(1, figsize=(8,4))
    semilogy(wfhist)
end;
```


![png](example_files/example_37_0.png)


The Wiener filter


```julia
let trn=trn, f=wfsim
    fig, ax = subplots(1, figsize=(8,4))
    x1, x2 = pix(trn)
    vm = extrema(f[:]) .|> abs |> x->max(x...)
    pcm = ax.pcolormesh(x2, x1, f[:],vmin=-vm, vmax=vm)
    ax.set_title("Wiener filter")
    fig.colorbar(pcm, ax = ax)
    fig.tight_layout()
end;
```


![png](example_files/example_39_0.png)

