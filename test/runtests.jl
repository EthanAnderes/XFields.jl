using XFields
using Test

@testset "XFields.jl" begin
    include("basic.jl")
    include("rFFT_impulse_test_1d.jl")
end
