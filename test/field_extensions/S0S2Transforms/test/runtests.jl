using XFields
using Test

include(joinpath(XFields.module_dir, "test/field_extensions/S0S2Transforms/src/S0S2Transforms.jl"))
using .S0S2Transforms


@testset "basic S0S2Transform" begin
    


	let nᵢ=(256, 50), pᵢ=(1.0, 0.5)
		ST = S0S2transform{nᵢ, pᵢ}
		FT = rFFT(ST)

		matx = rand(Float64, Grid(FT).nxi...,3)
		matk = rand(Float64, Grid(FT).nki...,3)

		@inferred ST * matx
		@inferred ST \ matk

		f = Smap{ST}(matx)
		g = Sfourier{ST}(matk)
		f + g

		f[:Ix]
		f[:El]
		f[:]
		f[!]

		w = Smap{ST}(matx[:,:,1], 1, 0)
		u = Sfourier{ST}(0,1, matk[:,:,3])

		DiagOp(w) * f
		DiagOp(w) * g
		DiagOp(u) * f
		DiagOp(u) * g
	end


	let nᵢ=(128, 128), Δxᵢ=(1/128, 1/128)
		ST = S0S2transform(nᵢ=nᵢ,Δxᵢ=Δxᵢ) 
		f = Smap{ST}()
		g = Sfourier{ST}()


		f + g

		f[:Ix]
		f[:El]
		f[:]
		f[!]

		w = Smap{ST}(matx[:,:,1], 1, 0)
		u = Sfourier{ST}(0,1, matk[:,:,3])

		DiagOp(w) * f
		DiagOp(w) * g
		DiagOp(u) * f
		DiagOp(u) * g
	end





end


