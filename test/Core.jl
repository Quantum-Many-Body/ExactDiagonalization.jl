using ExactDiagonalization
using ExactDiagonalization: SectorFilter
using QuantumLattices: Algorithm, Fock, Heisenberg, Hilbert, Hopping, Hubbard, Lattice, Onsite, Operator, OperatorSum, OperatorIndexToTuple, Parameters, Spin, Table
using QuantumLattices: contenttoconfig, getcontent, idtype, parameternames, showasleaf, 𝕔
using SparseArrays: SparseMatrixCSC

@testset "EDMatrix & EDEigenData" begin
    m = EDMatrix(SparseMatrixCSC(6, 6, [1, 3, 6, 9, 11, 14, 16], [1, 2, 1, 3, 4, 2, 3, 5, 2, 5, 3, 4, 6, 5, 6], [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]), BinaryBases(1:4, ℕ(2)))
    @test m == EDMatrix(m.matrix, m.bra, m.ket) == EDMatrix(m.matrix, (m.bra, m.ket))
    @test parameternames(typeof(m)) == (:value, :sector)
    @test getcontent(m, :id) == (m.bra, m.ket)
    @test getcontent(m, :value) == m.matrix
    @test scalartype(m) == scalartype(typeof(m)) == Float64
    @test promote_type(typeof(m), ComplexF64) == EDMatrix{SparseMatrixCSC{ComplexF64, Int}, BinaryBases{ℕ, BinaryBasis{UInt}, Vector{BinaryBasis{UInt}}}}
    @test showasleaf(typeof(m)) == false
    @test string(m) == "EDMatrix\n  matrix: 6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 15 stored entries:\n   2.0  1.0   ⋅    ⋅    ⋅    ⋅ \n   1.0   ⋅   1.0  1.0   ⋅    ⋅ \n    ⋅   1.0  2.0   ⋅   1.0   ⋅ \n    ⋅   1.0   ⋅    ⋅   1.0   ⋅ \n    ⋅    ⋅   1.0  1.0   ⋅   1.0\n    ⋅    ⋅    ⋅    ⋅   1.0  2.0\n  bra: 2^[1 2 3 4] => ℕ(2)\n  ket: 2^[1 2 3 4] => ℕ(2)"

    e = -1.8557725066359907
    v = [0.1373584690371194, -0.5296230084669367, 0.274716938074239, 0.5707844108834217, -0.5296230084669369, 0.13735846903711957]
    values, vectors = eigen(m)
    @test isapprox(values[1], e; atol=10^-10)
    @test isapprox(vectors[1], v; atol=10^-10) || isapprox(vectors[1], -v; atol=10^-10)
    values, vectors, sectors = eigen(OperatorSum(m))
    @test isapprox(values[1], e; atol=10^-10)
    @test isapprox(vectors[1], v; atol=10^-10) || isapprox(vectors[1], -v; atol=10^-10)
    @test sectors[1] == BinaryBases(1:4, ℕ(2))
end

@testset "EDMatrixization & SectorFilter" begin
    indexes = [𝕔(i, 1, 0, [0.0, 0.0], [0.0, 0.0]) for i = 1:4]
    table = Table(indexes, OperatorIndexToTuple(:site, :orbital, :spin))
    op₁, op₂, op₃ = Operator(2.0, indexes[2]', indexes[1]), Operator(2.0, indexes[3]', indexes[2]), Operator(2.0, indexes[4]', indexes[3])
    ops = op₁ + op₂ + op₃
    sectors = (BinaryBases(1:4, ℕ(1)), BinaryBases(1:4, ℕ(2)), BinaryBases(1:4, ℕ(3)))
    M = EDMatrix{SparseMatrixCSC{Float64, Int}, BinaryBases{ℕ, BinaryBasis{UInt}, Vector{BinaryBasis{UInt}}}}

    mr = EDMatrixization{Float64}(table, sectors...)
    @test valtype(typeof(mr), eltype(ops)) == valtype(typeof(mr), typeof(ops)) == OperatorSum{M, idtype(M)}
    @test contenttoconfig(mr) == (table, [(id(sectors[1]), id(sectors[1])), (id(sectors[2]), id(sectors[2])), (id(sectors[3]), id(sectors[3]))])

    ms = mr(ops)
    mr₁ = EDMatrixization{Float64}(table, sectors[1])
    mr₂ = EDMatrixization{Float64}(table, sectors[2])
    mr₃ = EDMatrixization{Float64}(table, sectors[3])
    @test ms == mr₁(ops) + mr₂(ops) + mr₃(ops)
    @test mr₁(ops) == mr₁(op₁) + mr₁(op₂) + mr₁(op₃)
    @test mr₂(ops) == mr₂(op₁) + mr₂(op₂) + mr₂(op₃)
    @test mr₃(ops) == mr₃(op₁) + mr₃(op₂) + mr₃(op₃)

    sf = SectorFilter(sectors[1])
    @test sf == SectorFilter((sectors[1], sectors[1]))
    @test valtype(typeof(sf), typeof(ms)) == typeof(ms)
    @test sf(ms) == mr₁(ops)
end

@testset "Binary ED" begin
    lattice = Lattice([0.0], [1.0])
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, 1.0, 1)
    U = Hubbard(:U, 0.0)
    μ = Onsite(:μ, 0.0)

    ed = Algorithm(Symbol("two-site"), ED(lattice, hilbert, (t, U, μ), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
    @test kind(ed.frontend) == kind(typeof(ed.frontend)) == EDKind(:Binary)
    @test scalartype(ed) == scalartype(ed.frontend) == scalartype(typeof(ed.frontend)) == Float64
    @test Parameters(ed) == (t=1.0, U=0.0, μ=0.0)
    @test contenttoconfig(ed.frontend) == (ed.frontend.lattice, contenttoconfig(ed.frontend.system), contenttoconfig(ed.frontend.matrixization))

    vector = [0.5, -0.5, -0.5, 0.5]
    eigensystem = eigen(matrix(prepare!(ed)); nev=1)
    @test count(eigensystem) == 1
    values, vectors, sectors = eigensystem
    @test values==eigensystem.values && vectors==eigensystem.vectors && sectors==eigensystem.sectors
    @test isapprox(eigensystem.values[1], -2.0; atol=10^-10)
    @test isapprox(eigensystem.vectors[1], vector; atol=10^-10) || isapprox(eigensystem.vectors[1], -vector; atol=10^-10)
    eigensystem = eigen(ed; nev=1)
    @test isapprox(eigensystem.values[1], -2.0; atol=10^-10)
    @test isapprox(eigensystem.vectors[1], vector; atol=10^-10) || isapprox(eigensystem.vectors[1], -vector; atol=10^-10)

    vector = [-0.43516214649359913, 0.5573454101893041, 0.5573454101893037, -0.43516214649359913]
    update!(release!(ed); U=1.0, μ=-0.5)
    eigensystem = eigen(ed, ℕ(length(lattice)) ⊠ 𝕊ᶻ(0); nev=1)
    @test isapprox(eigensystem.values[1], -2.5615528128088303; atol=10^-10)
    @test isapprox(eigensystem.vectors[1], vector; atol=10^-10) || isapprox(eigensystem.vectors[1], -vector; atol=10^-10)

    vector = [-0.37174803446018434, 0.6015009550075453, 0.6015009550075459, -0.3717480344601846]
    update!(ed; U=2.0, μ=-1.0)
    eigensystem = eigen(ed, ℕ(length(lattice)) ⊠ 𝕊ᶻ(0); nev=1)
    @test isapprox(eigensystem.values[1], -3.23606797749979; atol=10^-10)
    @test isapprox(eigensystem.vectors[1], vector; atol=10^-10) || isapprox(eigensystem.vectors[1], -vector; atol=10^-10)

    another = ED(ed.frontend.system, ℕ(length(lattice)) ⊠ 𝕊ᶻ(0))
    eigensystem = eigen(ed, ℕ(length(lattice)) ⊠ 𝕊ᶻ(0); nev=1)
    @test isapprox(eigensystem.values[1], -3.23606797749979; atol=10^-10)
    @test isapprox(eigensystem.vectors[1], vector; atol=10^-10) || isapprox(eigensystem.vectors[1], -vector; atol=10^-10)

    eigensystem = ed(:eigen, EDEigen(ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)); nev=1)
    @test isapprox(eigensystem.data.values[1], -3.23606797749979; atol=10^-10)
    @test isapprox(eigensystem.data.vectors[1], vector; atol=10^-10) || isapprox(eigensystem.data.vectors[1], -vector; atol=10^-10)
end

@testset "Abelian ED" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (4, 4))
    hilbert = Hilbert(site=>Spin{1//2}() for site=1:length(lattice))

    ed = ED(lattice, hilbert, Heisenberg(:J, 1.0, 1), (𝕊ᶻ(0), 𝕊ᶻ(1), 𝕊ᶻ(-1)))
    eigensystem = eigen(ed; nev=4)
    @test isapprox(eigensystem.values, [-9.189207065192935, -8.686937479074416, -8.686937479074407, -8.686937479074404]; atol=10^-12)

    another = ED(lattice, hilbert, Heisenberg(:J, 1.0, 1))
    eigensystem = eigen(another; nev=6)
    @test isapprox(eigensystem.values[1:4], [-9.189207065192946, -8.686937479074421, -8.686937479074418, -8.68693747907441]; atol=10^-12)

    third = ED(another.system)
    eigensystem = eigen(third; nev=6)
    @test isapprox(eigensystem.values[1:4], [-9.189207065192946, -8.686937479074421, -8.686937479074418, -8.68693747907441]; atol=10^-12)
end
