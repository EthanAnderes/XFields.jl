# tests


using XFields
using Test

include(joinpath(XFields.module_dir, "test/field_extensions/S0S2Fields1/src/S0S2Fields1.jl"))
import .S0S2Fields1



ni   = (256,256)     
Δxi  = (deg2rad(1/60), deg2rad(1/60))
FT   = rFFT(nᵢ=ni, Δxᵢ=Δxi)  # define Fourier transform
gr   = Grid(FT)                # get corresponding coordinate grid object

w = randn(gr.nxi...,3) ./ sqrt(gr.Ωx)

FT3 = plan(FT,XFields.LastDimSize{3})

sin2ϕ, cos2ϕ = S0S2Fields1.sin2ϕ_cos2ϕ(S0S2Fields1.S0S2fourier{FT})

function ŋs(::Type{FT}) where FT<:FourierTransform
	gr = Grid(FT)
	w = randn(gr.nxi...,3) ./ sqrt(gr.Ωx)
	return S0S2Fields1.S0S2map{FT}(w)
end

m1 = ŋs(FT)
m2 = ŋs(FT)
f1 = S0S2Fields1.S0S2fourier{FT}(ŋs(FT))
f3 = ŋs(FT) |> S0S2Fields1.S0S2fourier{FT}


