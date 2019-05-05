using Test

@testset "all" begin
    @testset "FED" begin include("FED.jl") end
end
