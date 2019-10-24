

##======================================================
pᵢ  = (1.0, 0.5) # periods
nᵢ   = (256, 256) # number of samples (left endpoint included)
FT  = rFFT{nᵢ,pᵢ,length(nᵢ)}
UFT = rFFTunitary{nᵢ,pᵢ,length(nᵢ)}

FT_plan = plan(FT)
UFT_plan = plan(UFT)

grid   = Grid(FT)
λ      = wavenumber(FT)
kfulli = frequencies(FT) 
xfulli = pixels(FT) 

fk = FT * rand(grid.nxi...)
fx = FT \ rand(Complex{Float64}, grid.nki...)

f1 = Smap{FT}(fx)
f2 = Sfourier{FT}(fk)

f4 = Sfourier{FT}(f1)
f3 = Smap{FT}(f2)

f1 + f2
- 2 * f2
map(x -> sin.(cos.(x)), fielddata(f1)) |> x->Smap{FT}(x...)

L = DiagOp(f1)
L * f2

L1 = DiagOp(f1)
L2 = DiagOp(f2)

L1 * f2
L2 * L1 * f2
L2 * L1

##======================================================
pᵢ  = (1.0,) 
nᵢ   = (256,)
FT  = rFFT{nᵢ,pᵢ,length(nᵢ)}

grid   = Grid(FT)
λ      = wavenumber(FT)
kfulli = frequencies(FT) 
xfulli = pixels(FT) 

fk = FT * rand(grid.nxi...)
fx = FT \ rand(Complex{Float64}, grid.nki...)

f1 = Smap{FT}(fx)
f2 = Sfourier{FT}(fk)

f4 = Sfourier{FT}(f1)
f3 = Smap{FT}(f2)

f1 + f2
- 2 * f2
map(x -> sin.(cos.(x)), fielddata(f1)) |> x->Smap{FT}(x...)

L = DiagOp(f1)
L * f2


    
## =========================================

nᵢ  = (256, 50) # number of samples (left endpoint included)
pᵢ  = (1.0, 0.5) # periods
Δxᵢ = tuple((pn[1]/pn[2] for pn in zip(pᵢ,nᵢ))...)

FFT1 = rFFT{nᵢ,pᵢ,length(nᵢ)}
FFT2 = rFFT(nᵢ=nᵢ, pᵢ=pᵢ)
FFT3 = rFFT(nᵢ=nᵢ, Δxᵢ=Δxᵢ)
FFTu1 = rFFTunitary{nᵢ,pᵢ,length(nᵢ)}
FFTu2 = rFFTunitary(nᵢ=nᵢ, pᵢ=pᵢ)
FFTu3 = rFFTunitary(nᵢ=nᵢ, Δxᵢ=Δxᵢ)

@test FFT1 == FFT2 == FFT3
@test FFTu1 == FFTu2 == FFTu3

grid1 = Grid(FFT1)
grid2 = Grid(FFT2)
grid3 = Grid(FFT3)
gridu1 = Grid(FFTu1)
gridu2 = Grid(FFTu2)
gridu3 = Grid(FFTu3)

@test grid1 == grid2 == grid3
@test gridu1 == gridu2 == gridu3



## =========================================

pᵢ  = (1.0, 0.5) # periods
nᵢ   = (256, 256) # number of samples (left endpoint included)
FFT  = cFFT{nᵢ,pᵢ,length(nᵢ)}
UFT  = cFFTunitary{nᵢ,pᵢ,length(nᵢ)}
grid = Grid(FFT)

@inferred Grid(FFT)
@inferred wavenumber(FFT)
@inferred frequencies(FFT) 
@inferred pixels(FFT) 

fk = FFT * rand(Complex{Float64}, grid.nxi...)
fx = FFT \ rand(Complex{Float64}, grid.nki...)

@test grid.nxi == grid.nki

## =========================================

pᵢ  = (1.0, 0.5) # periods
nᵢ   = (256, 256) # number of samples (left endpoint included)
FFT  = rFFT{nᵢ,pᵢ,length(nᵢ)}
UFT  = rFFTunitary{nᵢ,pᵢ,length(nᵢ)}
grid = Grid(FFT)

@inferred Grid(FFT)
@inferred wavenumber(FFT)
@inferred frequencies(FFT) 
@inferred pixels(FFT) 

fk = FFT * rand(nᵢ...)
fx = FFT \ rand(Complex{Float64}, grid.nki...)

f1 = Smap{FFT}(fx)
f2 = Sfourier{FFT}(fk)
f4 = Sfourier{FFT}(f1)
f3 = Smap{FFT}(f2)

@inferred f1 + f2
@inferred - 2 * f2
map(x -> sin.(cos.(x)), fielddata(f1)) |> x->Smap{FFT}(x...)

L = DiagOp(f1)
L * f2

L1 = DiagOp(f1)
L2 = DiagOp(f2)
for f ∈ (f1, f2), L ∈ (L1, L2), M ∈ (L1, L2)
	@inferred L * f
	@inferred L * M * f
	@inferred L * M \ f
	@inferred L \ M \ f
	@inferred L \ M * f
	@test sum(sum(abs2.(t)) for t in fielddata((L * M * f) - (L * (M * f)))) ≈ 0.0
	@test sum(sum(abs2.(t)) for t in fielddata((L * M \ f) - (inv(M) * (inv(L) * f)))) ≈ 0.0
	@test sum(sum(abs2.(t)) for t in fielddata((L \ M \ f) - (inv(M) * (inv(inv(L)) * f)))) ≈ 0.0
	@test sum(sum(abs2.(t)) for t in fielddata((L \ M * f) - (inv(L) * (M * f)))) ≈ 0.0
end


## =========================================

pᵢ  = (1.0,) 
nᵢ   = (256,)
FFT  = rFFT{nᵢ,pᵢ,length(nᵢ)}
UFT  = rFFTunitary{nᵢ,pᵢ,length(nᵢ)}
grid = Grid(FFT)

@inferred Grid(FFT)
@inferred wavenumber(FFT)
@inferred frequencies(FFT) 
@inferred pixels(FFT) 


fk = FFT * rand(nᵢ...)
fx = FFT \ rand(Complex{Float64}, grid.nki...)

f1 = Smap{FFT}(fx)
f2 = Sfourier{FFT}(fk)
f4 = Sfourier{FFT}(f1)
f3 = Smap{FFT}(f2)

@inferred f1 + f2
@inferred - 2 * f2
map(x -> sin.(cos.(x)), fielddata(f1)) |> x->Smap{FFT}(x...)

L = DiagOp(f1)
L * f2

L1 = DiagOp(f1)
L2 = DiagOp(f2)
for f ∈ (f1, f2), L ∈ (L1, L2), M ∈ (L1, L2)
	@inferred L * f
	@inferred L * M * f
	@inferred L * M \ f
	@inferred L \ M \ f
	@inferred L \ M * f
	@test sum(sum(abs2.(t)) for t in fielddata((L * M * f) - (L * (M * f)))) ≈ 0.0
	@test sum(sum(abs2.(t)) for t in fielddata((L * M \ f) - (inv(M) * (inv(L) * f)))) ≈ 0.0
	@test sum(sum(abs2.(t)) for t in fielddata((L \ M \ f) - (inv(M) * (inv(inv(L)) * f)))) ≈ 0.0
	@test sum(sum(abs2.(t)) for t in fielddata((L \ M * f) - (inv(L) * (M * f)))) ≈ 0.0
end


