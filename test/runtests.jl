using Test
using SafeTestsets

@safetestset "ExactDiagonalization" begin
    @time @safetestset "EDWithBinaryBases" begin include("EDWithBinaryBases.jl") end
    @time @safetestset "EDWithAbelianBases" begin include("EDWithAbelianBases.jl") end
end
