using Test
using LinearAlgebra: eigen
using SparseArrays: SparseMatrixCSC
using QuantumLattices: PID, Point, Lattice, FID, Index, OID, Table, OIDToTuple, Metric, Hilbert, Spin, Fock
using QuantumLattices: Operator, OperatorSum, FockTerm, SpinTerm, Hopping, Onsite, Hubbard, Parameters
using QuantumLattices: ⊗, ⊕, dimension, contentnames, getcontent, parameternames, dtype, add!, matrix, idtype, kind, statistics, update!
using ExactDiagonalization

@testset "BinaryBasis" begin
    basis = BinaryBasis(5)
    @test basis==BinaryBasis((1, 3)) && isequal(basis, BinaryBasis{UInt}((1, 3)))
    @test basis<BinaryBasis(6) && isless(basis, BinaryBasis(6))
    @test one(basis) == one(typeof(basis)) == BinaryBasis(1)
    @test zero(basis) == zero(typeof(basis)) == BinaryBasis(0)
    @test string(basis) == "101"
    @test eltype(basis) == eltype(typeof(basis)) == Int
    @test collect(basis) == [1, 3]
    @test one(basis, 2) == BinaryBasis(7)
    @test isone(basis, 1)==true && isone(basis, 2)==false && isone(basis, 3)==true
    @test zero(basis, 1) == BinaryBasis(4)
    @test iszero(basis, 1)==false && iszero(basis, 2)==true && iszero(basis, 3)==false
    @test count(basis, 1, 1)==1 && count(basis, 1, 2)==1 && count(basis, 1, 3)==2 && count(basis, 2, 3)==1
    @test basis==BinaryBasis([1])⊗BinaryBasis([3])
end

@testset "BinaryBasisRange" begin
    bbr = BinaryBasisRange(UInt(0):UInt(4))
    @test issorted(bbr) == true
    @test dimension(bbr) == 5
    for i = 1:dimension(bbr)
        @test bbr[i] == BinaryBasis(UInt(i-1))
    end
end

@testset "BinaryBases" begin
    bs = BinaryBases(2)
    @test issorted(bs) == true
    @test dimension(bs) == length(bs) == 4
    @test bs==BinaryBases(1:2)
    @test isequal(bs, BinaryBases((2, 1)))
    for i = 1:dimension(bs)
        @test bs[i]==BinaryBasis(i-1)
        @test findfirst(bs[i], bs) == i
    end
    @test eltype(bs) == eltype(typeof(bs)) == BinaryBasis{UInt}
    @test collect(bs) == map(BinaryBasis, [0, 1, 2, 3])
    @test repr(bs) == "2^2"
    @test string(bs) == "2^2:\n  0\n  1\n  10\n  11\n"

    bs = BinaryBases(4, 2)
    @test collect(bs) == map(BinaryBasis, [3, 5, 6, 9, 10, 12])
    @test repr(bs) == "C(4, 2)"
    for i = 1:dimension(bs)
        @test findfirst(bs[i], bs) == i
    end

    bs = BinaryBases(1:2, 1) ⊗ BinaryBases(3:4, 1)
    @test collect(bs) == map(BinaryBasis, [5, 6, 9, 10])
    @test repr(bs) == "C(2, 1) ⊗ C(2, 1)"
end

@testset "TargetSpace" begin
    bs₁ = BinaryBases(1:4, 2)
    bs₂ = BinaryBases(1:4, 3)
    ts = TargetSpace(bs₁, bs₂)
    @test contentnames(typeof(ts)) == (:table,)
    @test getcontent(ts, :table) == ts.sectors
    @test add!(add!(TargetSpace(typeof(bs₁)[]), bs₁), bs₂) == add!(TargetSpace(typeof(bs₁)[]), ts) == ts
    @test bs₁⊕bs₂ == ts

    bs₃ = BinaryBases(5:8, 2)
    @test ts⊕bs₃ == TargetSpace(bs₁, bs₂, bs₃)
    @test bs₃⊕ts == TargetSpace(bs₃, bs₁, bs₂)
    @test ts⊕TargetSpace(bs₃) == TargetSpace(bs₁, bs₂, bs₃)
end

@testset "matrix" begin
    oids = [OID(Index(PID(i), FID{:f}(1, 1, 1)), [0.0, 0.0], [0.0, 0.0]) for i = 1:4]
    table = Table(oids, OIDToTuple(:site, :orbital, :spin))

    braket = (BinaryBases(1:4, 2), BinaryBases(1:4, 3))
    ops = [Operator(2.0, oid) for oid in oids]
    @test matrix(ops[1], braket, table) == SparseMatrixCSC(6, 4, [1, 2, 3, 4, 4], [3, 5, 6], [2.0, 2.0, 2.0])
    @test matrix(ops[2], braket, table) == SparseMatrixCSC(6, 4, [1, 2, 3, 3, 4], [2, 4, 6], [-2.0, -2.0, 2.0])
    @test matrix(ops[3], braket, table) == SparseMatrixCSC(6, 4, [1, 2, 2, 3, 4], [1, 4, 5], [2.0, -2.0, -2.0])
    @test matrix(ops[4], braket, table) == SparseMatrixCSC(6, 4, [1, 1, 2, 3, 4], [1, 2, 3], [2.0, 2.0, 2.0])

    braket = (BinaryBases(1:4, 2), BinaryBases(1:4, 2))
    @test matrix(ops[2]'*ops[1], braket, table) == SparseMatrixCSC(6, 6, [1, 1, 2, 2, 3, 3, 3], [3, 5], [4.0, 4.0])
    @test matrix(ops[3]'*ops[1], braket, table) == SparseMatrixCSC(6, 6, [1, 2, 2, 2, 3, 3, 3], [3, 6], [-4.0, 4.0])
end

@testset "EDMatrix" begin
    m = EDMatrix(BinaryBases(1:4, 2), SparseMatrixCSC(6, 6, [1, 1, 2, 2, 3, 3, 3], [3, 5], [4.0, 4.0]))
    @test parameternames(typeof(m)) == (:sector, :value)
    @test getcontent(m, :id) == (m.bra, m.ket)
    @test getcontent(m, :value) == m.matrix
    @test dtype(m) == dtype(typeof(m)) == Float64
    @test promote_type(typeof(m), ComplexF64) == EDMatrix{BinaryBases{BinaryBasis{UInt64}, Vector{BinaryBasis{UInt64}}}, SparseMatrixCSC{ComplexF64, Int}}
end

@testset "EDMatrixRepresentation && SectorFilter" begin
    oids = [OID(Index(PID(i), FID{:f}(1, 1, 1)), [0.0, 0.0], [0.0, 0.0]) for i = 1:4]
    table = Table(oids, OIDToTuple(:site, :orbital, :spin))
    op₁, op₂, op₃ = Operator(2.0, oids[2]', oids[1]), Operator(2.0, oids[3]', oids[2]), Operator(2.0, oids[4]', oids[3])
    ops = op₁ + op₂ + op₃
    target = BinaryBases(1:4, 1)⊕BinaryBases(1:4, 2)⊕BinaryBases(1:4, 3)
    M = EDMatrix{BinaryBases{BinaryBasis{UInt64}, Vector{BinaryBasis{UInt64}}}, SparseMatrixCSC{Float64, Int}}

    mr = EDMatrixRepresentation(target, table)
    @test valtype(typeof(mr), eltype(ops)) == valtype(typeof(mr), typeof(ops)) == OperatorSum{M, idtype(M)}

    ms = mr(ops)
    mr₁ = EDMatrixRepresentation(TargetSpace(target[1]), table)
    mr₂ = EDMatrixRepresentation(TargetSpace(target[2]), table)
    mr₃ = EDMatrixRepresentation(TargetSpace(target[3]), table)
    @test ms == mr₁(ops) + mr₂(ops) + mr₃(ops)
    @test mr₁(ops) == mr₁(op₁) + mr₁(op₂) + mr₁(op₃)
    @test mr₂(ops) == mr₂(op₁) + mr₂(op₂) + mr₂(op₃)
    @test mr₃(ops) == mr₃(op₁) + mr₃(op₂) + mr₃(op₃)

    sf = SectorFilter(target[1])
    @test sf == SectorFilter((target[1], target[1]))
    @test valtype(typeof(sf), typeof(ms)) == typeof(ms)
    @test sf(ms) == mr₁(ops)
end

@testset "ED" begin
    @test EDKind(FockTerm) == EDKind(:FED)
    @test EDKind(SpinTerm) == EDKind(:SED)
    @test EDKind(Tuple{Hopping, Onsite, Hubbard}) == EDKind(:FED)

    @test Metric(EDKind(:FED), Hilbert(PID(1)=>Fock{:f}(1, 2, 2))) == OIDToTuple(:spin, :site, :orbital)
    @test Metric(EDKind(:SED), Hilbert(PID(1)=>Spin{1//2}(1))) == OIDToTuple(:site, :orbital)

    lattice = Lattice(:L2P, [Point(PID(1), [0.0]), Point(PID(2), [1.0])])
    hilbert = Hilbert(pid=>Fock{:f}(1, 2, 2) for pid in lattice.pids)
    t = Hopping(:t, 1.0, 1)
    U = Hubbard(:U, 0.0, modulate=true)
    μ = Onsite(:μ, 0.0, modulate=true)

    bases = BinaryBases(1:2, 1)⊗BinaryBases(3:4, 1)
    ed = ED(lattice, hilbert, (t, U, μ), TargetSpace(bases))
    @test kind(ed) == kind(typeof(ed)) == EDKind(:FED)
    @test eltype(ed) == eltype(typeof(ed)) == eltype(ed.H)
    @test valtype(ed) == valtype(typeof(ed)) == Float64
    @test statistics(ed) == statistics(typeof(ed)) == :f
    @test Parameters(ed) == (t=1.0, U=0.0, μ=0.0)

    eigensystem = eigen(matrix(ed); nev=1)
    @test isapprox(eigensystem.values, [-2.0]; atol=10^-10)
    @test isapprox(eigensystem.vectors, [0.5; -0.5; -0.5; 0.5]; atol=10^-10)

    update!(ed, U=1.0, μ=-0.5)
    eigensystem = eigen(matrix(ed); nev=1)
    @test isapprox(eigensystem.values, [-2.5615528128088303]; atol=10^-10)
    @test isapprox(eigensystem.vectors, [-0.43516214649359913; 0.5573454101893041; 0.5573454101893037; -0.43516214649359913]; atol=10^-10)
end
