using ExactDiagonalization
using LinearAlgebra: eigen
using QuantumLattices: Abelian, Graded, Heisenberg, Hilbert, Lattice, Metric, OperatorIndexToTuple, Spin, Table, 𝕊, 𝕊ᶻ, ℤ₁
using QuantumLattices: ⊠, dimension, id, matrix, partition, prepare!
using SparseArrays: SparseMatrixCSC

@testset "AbelianBases" begin
    @test partition(1) == (Int[], [1])
    @test partition(4) == ([1, 2], [3, 4])
    @test partition(5) == ([1, 2], [3, 4, 5])

    bs = AbelianBases([2, 2])
    @test id(bs) == (ℤ₁(0), [Graded{ℤ₁}(0=>2), Graded{ℤ₁}(0=>2)], ([1], [2]))
    @test dimension(bs) == 4
    @test string(bs) == "{[Graded{ℤ₁}(0=>2)₁] ⊗ [Graded{ℤ₁}(0=>2)₂]: ℤ₁(0)}"
    @test match(bs, bs)
    @test Abelian(bs) == ℤ₁(0)
    @test !sumable(bs, bs)

    locals = [Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1), Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)]
    another = AbelianBases(locals, 𝕊ᶻ(0))
    @test id(another) == (𝕊ᶻ(0), locals, ([1], [2]))
    @test dimension(another) == 2
    @test string(another) == "{[Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)₁] ⊗ [Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)₂]: 𝕊ᶻ(0)}"
    @test match(another, AbelianBases(locals, 𝕊ᶻ(1)))
    @test Abelian(another) == 𝕊ᶻ(0)
    @test sumable(another, AbelianBases(locals, 𝕊ᶻ(1)))
end

@testset "Graded Spin" begin
    spin = Spin{3//2}()
    @test Graded{ℤ₁}(spin) == Graded{ℤ₁}(0=>4)
    @test Graded{𝕊ᶻ}(spin) == Graded{𝕊ᶻ}(-3//2=>1, -1//2=>1, 1//2=>1, 3//2=>1)'
end

@testset "matrix SpinIndex" begin
    index = 𝕊{1//2}('y')
    @test matrix(index, Graded{ℤ₁}(0=>2)) == matrix(index)
end

@testset "EDKind & Metric" begin
    @test EDKind(Hilbert{Spin{1//2}}) == EDKind(:Abelian)
    @test Metric(EDKind(:Abelian), Hilbert(1=>Spin{1//2}())) == OperatorIndexToTuple(:site)
end

@testset "Sector" begin
    hilbert = Hilbert(Spin{1//2}(), 2)
    @test Sector(hilbert) == AbelianBases([2, 2])
    @test Sector(hilbert, 𝕊ᶻ(0)) == AbelianBases([Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)', Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)'], 𝕊ᶻ(0))
end

@testset "TargetSpace" begin
    hilbert = Hilbert(Spin{1//2}(), 2)
    table = Table(hilbert, Metric(EDKind(hilbert), hilbert))
    @test TargetSpace(hilbert, 𝕊ᶻ(0), table) ==  TargetSpace([AbelianBases([Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)', Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)'], 𝕊ᶻ(0))], table)
end

@testset "Abelian ED" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (4, 4))
    hilbert = Hilbert(site=>Spin{1//2}() for site=1:length(lattice))

    ed = ED(lattice, hilbert, Heisenberg(:J, 1.0, 1), (𝕊ᶻ(0), 𝕊ᶻ(1), 𝕊ᶻ(-1)); delay=true)
    eigensystem = eigen(matrix(prepare!(ed)); nev=4)
    @test isapprox(eigensystem.values, [-9.189207065192935, -8.686937479074416, -8.686937479074407, -8.686937479074404]; atol=10^-12)

    ed = ED(lattice, hilbert, Heisenberg(:J, 1.0, 1); delay=true)
    eigensystem = eigen(matrix(prepare!(ed)); nev=6)
    @test isapprox(eigensystem.values[1:4], [-9.189207065192946, -8.686937479074421, -8.686937479074418, -8.68693747907441]; atol=10^-12)
end

# @testset "spincoherentstates" begin
#     unitcell = Lattice([0.0, 0.0], [0.0, √3/3]; vectors=[[1.0, 0.0], [0.5, √3/2]])
#     lattice = Lattice(unitcell, (2, 2), ('P', 'P'))

#     spins = Dict(i=>(isodd(i) ? [0, 0, 1] : [0, 0, -1]) for i=1:length(lattice))
#     state = spincoherentstates(xyz2ang(spins))
#     @test findmax(state.|>abs) == (1.0, 171)

#     hilbert = Hilbert(Spin{1//2}(), length(lattice))
#     targetspace = TargetSpace(hilbert)

#     k, s = structure_factor(lattice, targetspace[1], hilbert, state)
#     @test isapprox(s[3, 2, 11], 1.418439381905401)
#     @test isapprox(structure_factor(lattice, targetspace[1], hilbert, state, [0.0, 4*pi/sqrt(3)])[3], 1.25)

#     sp = Dict(1:2:length(lattice)|>collect=>[0.0, 0.0], 2:2:length(lattice)|>collect=>[pi, 0.0])
#     seta, p, pscs = Pspincoherentstates(state, sp)
#     @test isapprox(pscs[:, 1], [1.0 for _=1:length(pscs[:, 1])])
# end
