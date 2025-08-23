using Test
using SafeTestsets

@safetestset "ExactDiagonalization" begin
    @time @safetestset "BandLanczos" begin include("BandLanczos.jl") end
    @time @safetestset "EDWithBinaryBases" begin include("EDWithBinaryBases.jl") end
    @time @safetestset "EDWithAbelianBases" begin include("EDWithAbelianBases.jl") end
end
