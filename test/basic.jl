


#%% Uniform scaling
#%% ------------------------------------------

let FT = rFFT(nᵢ=(256, 50), pᵢ=(1.0, 0.5))

	fmap = rand(Bool, Grid(FT).nxi...) |> Smap{FT}
	gfourier = rand(Bool, Grid(FT).nki...) |> Sfourier{FT}

	Dmap = DiagOp(fmap)
	Dfourier = DiagOp(fmap)

	Id2 = 2*LinearAlgebra.I

	@inferred Id2 * fmap 
	@inferred Id2 * gfourier
	@inferred Id2 \ fmap 
	@inferred Id2 \ gfourier

	@inferred Dmap * Id2
	@inferred Dfourier * Id2
	@inferred Dmap + Id2
	@inferred Dfourier + Id2
	@inferred Dmap - Id2
	@inferred Dfourier - Id2

	@inferred Id2 * Dmap
	@inferred Id2 * Dfourier
	@inferred Id2 + Dmap
	@inferred Id2 + Dfourier
	@inferred Id2 - Dmap
	@inferred Id2 - Dfourier 


	# @test Id2 * fmap 
	# @test Id2 * gfourier
	# @test Id2 \ fmap 
	# @test Id2 \ gfourier

	# @test Dmap * Id2
	# @test Dfourier * Id2
	# @test Dmap + Id2
	# @test Dfourier + Id2
	# @test Dmap - Id2
	# @test Dfourier - Id2

	# @test Id2 * Dmap
	# @test Id2 * Dfourier
	# @test Id2 + Dmap
	# @test Id2 + Dfourier
	# @test Id2 - Dmap
	# @test Id2 - Dfourier 




end


#%% SField Constructors
#%% ------------------------------------------

let FT = rFFT(nᵢ=(256, 50), pᵢ=(1.0, 0.5))

	@inferred Smap{FT}()
	@inferred Smap{FT}(0)
	@inferred Sfourier{FT}()
	@inferred Sfourier{FT}(0)

	matx = rand(Bool, Grid(FT).nxi...)
	matk = rand(Bool, Grid(FT).nki...)

	@inferred Smap{FT}(matx)
	@inferred Sfourier{FT}(matk)

end



#%% Test using FT for converting between Smap and Sfourier
#%% ------------------------------------------

let FT = rFFT(nᵢ=(256, 50), pᵢ=(1.0, 0.5))
	
	matx = rand(Float64, Grid(FT).nxi...)
	matk = FT*rand(Float64, Grid(FT).nxi...)
	fx = matx |> Smap{FT}
	fk = matk |> Sfourier{FT}

	@inferred FT * fx
	@inferred FT * fk
	@inferred FT \ fx
	@inferred FT * fk

	@test all((FT * fx).k .== (FT * matx))
	@test all((FT \ fx).x .== matx)

	@test all((FT * fk).k .== matk)
	@test all((FT \ fk).x .== (FT \ matk))


end





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

@test all(f1[!] .== FT_plan * fx)
@test all(f1[:] .== fx)
@test all(f2[!] .== fk)
@test all(f2[:] .== FT_plan \ fk)



@inferred dot(f3, f1)
# real(dot(cplan(FT) * f3[:x], cplan(FT) * f1[:x])) / Grid(FT).Ωk
# real(dot(f3[:x], f1[:x])) / Grid(FT).Ωx

f1 + f2
- 2 * f2

L = DiagOp(f1)
L * f2

L1 = DiagOp(f1)
L2 = DiagOp(f2)
L3 = sqrt(L2)

@test all(L1[!] .== FT_plan * fx)
@test all(L1[:] .== fx)
@test all(L2[!] .== fk)
@test all(L2[:] .== FT_plan \ fk)


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

@test grid1.ki == grid2.ki == grid3.ki
@test gridu1.ki == gridu2.ki == gridu3.ki



## =========================================

pᵢ  = (1.0, 0.5) # periods
nᵢ   = (256, 256) # number of samples (left endpoint included)
cFT  = cFFT{nᵢ,pᵢ,length(nᵢ)}
cUFT  = cFFTunitary{nᵢ,pᵢ,length(nᵢ)}
rFT  = rFFT{nᵢ,pᵢ,length(nᵢ)}
rUFT  = rFFTunitary{nᵢ,pᵢ,length(nᵢ)}

grid = Grid(cFT)

@inferred Grid(cFT)
@inferred wavenumber(cFT)
@inferred frequencies(cFT) 
@inferred pixels(cFT) 

cfx = rand(Complex{Float64}, grid.nxi...)
rfx = rand(Float64, grid.nxi...)


@test grid.nxi == grid.nki

@test sum(abs2, plan(cFT)  * cfx .- cplan(rFT)  * cfx) ≈ 0.0 
@test sum(abs2, plan(cUFT) * cfx .- cplan(rUFT) * cfx) ≈ 0.0 

@test sum(abs2, plan(cFT)  * cfx .- cplan(cFT)  * cfx) ≈ 0.0 
@test sum(abs2, plan(cUFT) * cfx .- cplan(cUFT) * cfx) ≈ 0.0 

@test sum(abs2, plan(rFT)  * rfx .- rplan(rFT)  * rfx) ≈ 0.0 
@test sum(abs2, plan(rUFT) * rfx .- rplan(rUFT) * rfx) ≈ 0.0 

@test sum(abs2, plan(rFT)  * rfx .- rplan(cFT)  * rfx) ≈ 0.0 
@test sum(abs2, plan(rUFT) * rfx .- rplan(cUFT) * rfx) ≈ 0.0 

@inferred adjoint(plan(cFT))
@inferred adjoint(plan(rFT))
@inferred transpose(plan(cFT))
@inferred transpose(plan(rFT))



## =========================================

pᵢ  = (1.0, 3.5) # periods
nᵢ   = (64,64) # number of samples (left endpoint included)
FFT  = rFFT{nᵢ,pᵢ,length(nᵢ)}
UFT  = rFFTunitary{nᵢ,pᵢ,length(nᵢ)}
grid = Grid(FFT)

@inferred Grid(FFT)
@inferred wavenumber(FFT)
@inferred frequencies(FFT) 
@inferred pixels(FFT) 

fk = FFT * randn(nᵢ...)
fx = FFT \ randn(Complex{Float64}, grid.nki...)

f1 = Smap{FFT}(fx)
f2 = Sfourier{FFT}(fk)
f4 = Sfourier{FFT}(f1)
f3 = Smap{FFT}(f2)

@inferred f1 + f2
@inferred - 2 * f2

L = DiagOp(f1)
L * f2

L1 = DiagOp(f1+1)
L2 = DiagOp(f2+1)
for f ∈ (f1, f2), L ∈ (L1, L2), M ∈ (L1, L2)
	@inferred L * f
	@inferred L * M * f
	@inferred L * M \ f
	@inferred L \ M \ f
	@inferred L \ M * f
	@test mean(mean(abs.(t)) for t in fielddata((L * M * f) - (L * (M * f)))) ≈ 0.0 atol=1e-5
	@test mean(mean(abs.(t)) for t in fielddata((L * M \ f) - (inv(M) * (inv(L) * f)))) ≈ 0.0 atol=1e-5
	@test mean(mean(abs.(t)) for t in fielddata((L \ M \ f) - (inv(M) * (inv(inv(L)) * f)))) ≈ 0.0 atol=1e-5
	@test mean(mean(abs.(t)) for t in fielddata((L \ M * f) - (inv(L) * (M * f)))) ≈ 0.0 atol=1e-5
end


## =========================================

pᵢ  = (1.0,) 
nᵢ   = (64,)
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

L = DiagOp(f1)
L * f2

L1 = DiagOp(f1+1)
L2 = DiagOp(f2+1)
for f ∈ (f1, f2), L ∈ (L1, L2), M ∈ (L1, L2)
	@inferred L * f
	@inferred L * M * f
	@inferred L * M \ f
	@inferred L \ M \ f
	@inferred L \ M * f
	@test mean(mean(abs.(t)) for t in fielddata((L * M * f) - (L * (M * f)))) ≈ 0.0 atol=1e-5
	@test mean(mean(abs.(t)) for t in fielddata((L * M \ f) - (inv(M) * (inv(L) * f)))) ≈ 0.0 atol=1e-5
	@test mean(mean(abs.(t)) for t in fielddata((L \ M \ f) - (inv(M) * (inv(inv(L)) * f)))) ≈ 0.0 atol=1e-5
	@test mean(mean(abs.(t)) for t in fielddata((L \ M * f) - (inv(L) * (M * f)))) ≈ 0.0 atol=1e-5
end


