using XFields
using LinearAlgebra
using FFTransforms
using Test

# mean(x) = sum(x) / length(x)

# @testset "basic" begin
#     include("basic.jl")
# end

# @testset "rFFTimpulses" begin
#     include("rFFTimpulses/test_get_rFFTimpulses_1.jl")
#     include("rFFTimpulses/test_get_rFFTimpulses_2.jl")
# end

@testset "Xfield constructors" begin

	let 
		Tf = Float32
		n1, n2 = 100, 256
		p2 = 2π 
		ft = 𝕀(n1) ⊗ 𝕎(Tf,n2,p2)
		
		fx = randn(Tf, size_in(ft))
		fk = plan(ft) * fx
		gk = plan(ft) * randn(Tf, size_in(ft))
		gx = plan(ft) \ gk

		fmap     = @inferred Xmap(ft, fx)
		ffourier = @inferred Xfourier(ft, gk)

		@inferred ft * fmap
		@inferred ft \ fmap
		@inferred ft * ffourier
		@inferred ft \ ffourier

		@test sum(abs2, fmap[:] .- fx) ≈ 0
		@test sum(abs2, fmap[!] .- fk) ≈ 0
		@test sum(abs2, ffourier[!] .- gk) ≈ 0
		@test sum(abs2, ffourier[:] .- gx) ≈ 0
	end


	let 
		Tf = Complex{Float64}
		n1, n2 = 111, 256
		ft = 𝕎(Tf,n1,2π) ⊗ 𝕀(n2) 
		
		fx = rand(Bool, size_in(ft))
		fk = plan(ft) * Tf.(fx)
		gk = rand(Bool, size_out(ft))
		Ti = eltype_out(ft)
		gx = plan(ft) \ (Ti.(gk))

		fmap     = @inferred Xmap(ft, fx)
		gfourier = @inferred Xfourier(ft, gk)

		@test sum(abs2, fmap[:] .- fx) ≈ 0
		@test sum(abs2, fmap[!] .- fk) ≈ 0
		@test sum(abs2, gfourier[!] .- gk) ≈ 0
		@test sum(abs2, gfourier[:] .- gx) ≈ 0
	end




	let 
		Tf = Complex{Float64}
		n1, n2 = 100, 256
		p1, p2 = 1.0, 2π 
		ft = 𝕎(Tf,n1,p1) ⊗ 𝕎(Tf,n2,p2)		
		
		ax = randn(Tf, size_in(ft))
		ak = plan(ft) * ax
		amap = @inferred Xmap(ft, ax)
		bmap = @inferred Xmap(ft,1)
		cmap = @inferred Xmap(ft)

		dk = plan(ft) * randn(Tf, size_in(ft))
		dx = plan(ft) \ dk
		dfourier = @inferred Xfourier(ft, dk)
		efourier = @inferred Xfourier(ft, 1)
		ffourier = @inferred Xfourier(ft)

		@test sum(abs2, amap[:] .- ax) ≈ 0
		@test sum(abs2, amap[!] .- ak) ≈ 0
		@test sum(abs2, bmap[:] .- 1) ≈ 0
		@test sum(abs2, cmap[:] .- 0) ≈ 0

		@test sum(abs2, dfourier[!] .- dk) ≈ 0
		@test sum(abs2, dfourier[:] .- dx) ≈ 0
		@test sum(abs2, efourier[!] .- 1) ≈ 0
		@test sum(abs2, ffourier[!] .- 0) ≈ 0
	end



end




@testset "Xfield DiagOp" begin

	let 
		pd  = (1.0, 3.5)
		sz  = (64,28)
		Tf  = Float64
		Ti  = Complex{Tf} 
		ft  = 𝕎(Tf,sz,pd)
		F   = typeof(ft)

		Xf = Xfourier{F, Tf, Ti, 2}
		Xm = Xmap{F, Tf, Ti, 2}

		fm = @inferred Xm(ft,rand(Tf,sz))
		ff = @inferred Xf(Xm(ft,rand(Tf,sz)))

		@inferred Xfourier(fm)
		@inferred Xf(fm)
		@inferred Xmap(fm)
		@inferred Xm(fm)

		@inferred Xfourier(ff)
		@inferred Xf(ff)
		@inferred Xmap(ff)
		@inferred Xm(ff)

		@inferred fm + ff
		@inferred - 2 * ff

		L = DiagOp(fm)
		L.f * Xmap(ff)
		ff isa Xf
		L * ff

		L1 = DiagOp(fm+1)
		L2 = DiagOp(ff+1)



		L1 * L2
		L2 * L2
		L1 * L1

		L2 * L2 * fm
		L2 * (L2 * fm)

		L1 * L1 * ff
		L1 * (L1 * ff)

		L1 * L2 * L2

		L1 * L2 \ L2
		
		for f ∈ (fm, ff), L ∈ (L1, L2), M ∈ (L1, L2)
			@inferred L * f
			@inferred L * M * f
			@inferred L * M \ f
			@inferred L \ M \ f
			@inferred L \ M * f
			@test sum(abs2, fielddata((L * M * f) - (L * (M * f)))) ≈ 0.0 atol=1e-5
			@test sum(abs2, fielddata((L * M \ f) - (inv(M) * (inv(L) * f)))) ≈ 0.0 atol=1e-5
			@test sum(abs2, fielddata((L \ M \ f) - (inv(M) * (inv(inv(L)) * f)))) ≈ 0.0 atol=1e-5
			@test sum(abs2, fielddata((L \ M * f) - (inv(L) * (M * f)))) ≈ 0.0 atol=1e-5
		end

	end
end




@testset "Uniform scaling" begin

	let nᵢ=(256, 50), pᵢ=(1.0, 0.5), Tf  = Float64

		Ti  = Complex{Tf} 
		ft  = 𝕎(Tf,nᵢ,pᵢ)


		fmap    = Xmap(ft, rand(Bool, size_in(ft)))
		gfourier = Xfourier(ft, rand(Bool, size_out(ft)))

		Dmap = DiagOp(fmap)
		Dfourier = DiagOp(gfourier)

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

		# TODO add actual tests here ..
	end

end









#=

#' SField Constructors
#' ------------------------------------------

let FT = rFFT(nᵢ=(256, 50), pᵢ=(1.0, 0.5))

	@inferred Rmap{FT}()
	@inferred Rmap{FT}(0)
	@inferred Rfourier{FT}()
	@inferred Rfourier{FT}(0)

	matx = rand(Bool, Grid(FT).nxi...)
	matk = rand(Bool, Grid(FT).nki...)

	@inferred Rmap{FT}(matx)
	@inferred Rfourier{FT}(matk)

end



#' Test using FT for converting between Rmap and Rfourier
#' ------------------------------------------

let FT = rFFT(nᵢ=(256, 50), pᵢ=(1.0, 0.5))
	
	matx = rand(Float64, Grid(FT).nxi...)
	matk = FT*rand(Float64, Grid(FT).nxi...)
	fx = matx |> Rmap{FT}
	fk = matk |> Rfourier{FT}

	@inferred FT * fx
	@inferred FT * fk
	@inferred FT \ fx
	@inferred FT * fk

	@test all((FT * fx).k .== (FT * matx))
	@test all((FT \ fx).x .== matx)

	@test all((FT * fk).k .== matk)
	@test all((FT \ fk).x .== (FT \ matk))


end





#' ======================================================
pᵢ  = (1.0, 0.5) # periods
nᵢ   = (256, 256) # number of samples (left endpoint included)
FT  = rFFT{Float32,nᵢ,pᵢ,length(nᵢ)}
UFT = rFFTunitary{Float32,nᵢ,pᵢ,length(nᵢ)}

FT_plan = plan(FT)
UFT_plan = plan(UFT)


grid   = Grid(FT)
λ      = wavenumber(FT)
kfulli = frequencies(FT) 
xfulli = pixels(FT) 

fk = FT * rand(grid.nxi...)
fx = FT \ rand(Complex{Float32}, grid.nki...)

f1 = Rmap{FT}(fx)
f2 = Rfourier{FT}(fk)
f4 = Rfourier{FT}(f1)
f3 = Rmap{FT}(f2)

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

#' ======================================================
pᵢ  = (1.0,) 
nᵢ   = (256,)
FT  = rFFT{Float64,nᵢ,pᵢ,length(nᵢ)}

grid   = Grid(FT)
λ      = wavenumber(FT)
kfulli = frequencies(FT) 
xfulli = pixels(FT) 

fk = FT * rand(grid.nxi...)
fx = FT \ rand(Complex{Float64}, grid.nki...)

f1 = Rmap{FT}(fx)
f2 = Rfourier{FT}(fk)

f4 = Rfourier{FT}(f1)
f3 = Rmap{FT}(f2)

f1 + f2
- 2 * f2

L = DiagOp(f1)
L * f2


    
#' =========================================

nᵢ  = (256, 50) # number of samples (left endpoint included)
pᵢ  = (1.0, 0.5) # periods
Δxᵢ = tuple((pn[1]/pn[2] for pn in zip(pᵢ,nᵢ))...)

FFT1 = rFFT{Float64,nᵢ,pᵢ,length(nᵢ)}
FFT2 = rFFT(nᵢ=nᵢ, pᵢ=pᵢ)
FFT3 = rFFT(nᵢ=nᵢ, Δxᵢ=Δxᵢ)
FFTu1 = rFFTunitary{Float64,nᵢ,pᵢ,length(nᵢ)}
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



#' =========================================

pᵢ  = (1.0, 0.5) # periods
nᵢ   = (256, 256) # number of samples (left endpoint included)
cFT  = cFFT{Float64,nᵢ,pᵢ,length(nᵢ)}
cUFT  = cFFTunitary{Float64,nᵢ,pᵢ,length(nᵢ)}
rFT  = rFFT{Float64,nᵢ,pᵢ,length(nᵢ)}
rUFT  = rFFTunitary{Float64,nᵢ,pᵢ,length(nᵢ)}

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




=#