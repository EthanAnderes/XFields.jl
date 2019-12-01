using XFields
using Test

mean(x) = sum(x) / length(x)

@testset "XFields.jl" begin
    include("basic.jl")
    include("test_get_rFFTimpulses.jl")
end

