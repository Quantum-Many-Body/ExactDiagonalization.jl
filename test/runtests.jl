using Test
using SafeTestsets

@safetestset "ExactDiagonalization" begin
    @time @safetestset "BandLanczos" begin include("BandLanczos.jl") end
    @time @safetestset "QuantumNumbers" begin include("QuantumNumbers.jl") end
    @time @safetestset "BinaryBases" begin include("BinaryBases.jl") end
    @time @safetestset "AbelianBases" begin include("AbelianBases.jl") end
    @time @safetestset "Core" begin include("Core.jl") end
    @time @safetestset "GreenFunctions" begin include("GreenFunctions.jl") end
    @time @safetestset "Assignments" begin include("Assignments.jl") end
end
