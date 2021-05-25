using Test
using Arpack: eigs
using ExactDiagonalization.FED
using SparseArrays: SparseMatrixCSC
using QuantumLattices
using QuantumLattices.Mathematics.AlgebraOverFields: idtype

@testset "GBasis" begin
    basis = GBasis{UInt}(2)
    @test kind(basis) == kind(typeof(basis)) == :g
    @test idtype(basis) == idtype(typeof(basis)) == Int
    @test dimension(basis) == 2^2
    @test id(basis) == 2
    @test basis[4] == 3
    @test searchsortedfirst(basis, 3) == 4
    @test string(basis) == "GBasis(2):\n  0\n  1\n  10\n  11\n"
    @test repr(basis) == "GBasis(2)"
end

@testset "SparseMatrixCSC" begin
    lattice = Lattice("L2P", [Point(PID(1), [0.0]), Point(PID(2), [1.0])])
    config = Config{Fock{:f}}(pid->Fock{:f}(atom=pid.site%2, norbital=1, nspin=2, nnambu=2), lattice.pids)
    table = Table(config, usualfockindextotuple)
    t = Hopping(:t, 1.0, 1)
    U = Hubbard(:U, 1.0)
    μ = Onsite(:μ, -0.5)
    g = Generator((t, U, μ), Bonds(lattice), config, half=false, table=table)
    m = SparseMatrixCSC(expand(g), GBasis{UInt}(4), g.table)
    @test eigs(m, which=:SR, ritzvec=false)[1][1] ≈ -2.5615528128088303
end
