# tests


using XFields
using Test

include(joinpath(XFields.module_dir, "test/field_extensions/S0S2Fields2/src/S0S2Fields2.jl"))
using .S0S2Fields2



ni   = (256,256)     
Δxi  = (deg2rad(1/60), deg2rad(1/60))
FT   = rFFT(nᵢ=ni, Δxᵢ=Δxi)  # define Fourier transform
gr   = Grid(FT)                # get corresponding coordinate grid object

sin2ϕ, cos2ϕ = S0S2Fields2.sin2ϕ_cos2ϕ(S0S2fourier{FT})

function ŋs(::Type{FT}) where FT<:FourierTransform
	gr = Grid(FT)
	w = randn(gr.nxi...,3) ./ sqrt(gr.Ωx)
	return S0S2Fields2.S0S2map{FT}(w)
end

m1 = ŋs(FT)
m2 = ŋs(FT)
f1 = S0S2Fields2.S0S2fourier{FT}(ŋs(FT))
f2 = ŋs(FT)

f1 + f2
f1[:Ix]
f1[:Bl]


