using XFields
using Test

# @testset "CMBrings.jl" begin
#     # Write your tests here.
# end


using FFTransforms

ft = 𝕀(512) ⊗ r𝕎32(4096,2π)

qu1 = Xmap(ft,     randn(eltype_in(ft),size_in(ft)))
qu2 = Xfourier(ft, randn(eltype_out(ft),size_out(ft)))

qu1 + qu2