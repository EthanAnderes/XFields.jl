using XFields
using Test

mean(x) = sum(x) / length(x)

@testset "basic" begin
    include("basic.jl")
end

@testset "rFFTimpulses" begin
    include("rFFTimpulses/test_get_rFFTimpulses_1.jl")
    include("rFFTimpulses/test_get_rFFTimpulses_2.jl")
end
