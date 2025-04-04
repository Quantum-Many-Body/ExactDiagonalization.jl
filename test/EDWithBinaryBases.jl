using ExactDiagonalization
using QuantumLattices: Abelian, Algorithm, Fock, Hilbert, Hopping, Hubbard, Lattice, Metric, Onsite, Operator, OperatorSum, OperatorIndexToTuple, Parameters, Table, 𝕔, ℕ, 𝕊ᶻ, ℤ₁
using QuantumLattices: ⊕, ⊗, ⊠, add!, dimension, getcontent, id, idtype, kind, matrix, parameternames, scalartype, update!
using SparseArrays: SparseMatrixCSC

@testset "BinaryBasis" begin
    @test basistype(Int8(1)) == basistype(UInt8) == UInt8
    @test basistype(Int16(1)) == basistype(UInt16) == UInt16
    @test basistype(Int32(1)) == basistype(UInt32) == UInt32
    @test basistype(Int64(1)) == basistype(UInt64) == UInt64
    @test basistype(Int128(1)) == basistype(UInt128) == UInt128

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
    @test count(basis)==2 && count(basis, 1, 1)==1 && count(basis, 1, 2)==1 && count(basis, 1, 3)==2 && count(basis, 2, 3)==1
    @test basis == BinaryBasis([1]) ⊗ BinaryBasis([3])
end

@testset "BinaryBasisRange" begin
    bbr = BinaryBasisRange(2)
    @test issorted(bbr) == true
    @test length(bbr) == 4
    for i = 1:length(bbr)
        @test bbr[i] == BinaryBasis(UInt(i-1))
    end
end

@testset "BinaryBases" begin
    bs = BinaryBases(2)
    @test bs == BinaryBases(1:2)
    @test isequal(bs, BinaryBases((2, 1)))
    @test hash(bs, UInt(1)) == hash(id(bs), UInt(1)) == hash((bs.quantumnumbers, bs.stategroups), UInt(1))
    @test dimension(bs) == length(bs) == 4
    @test bs[begin]==bs[1] && bs[end]==bs[4]
    for i = 1:length(bs)
        @test bs[i]==BinaryBasis(i-1)
        @test searchsortedfirst(bs[i], bs) == i
    end
    @test eltype(bs) == eltype(typeof(bs)) == BinaryBasis{UInt}
    @test collect(bs) == map(BinaryBasis, [0, 1, 2, 3])
    @test string(bs) == "{2^[1 2]: ℤ₁(0)}"
    @test Abelian(bs) == ℤ₁(0)

    bs = BinaryBases(4, ℕ(2))
    @test collect(bs) == map(BinaryBasis, [3, 5, 6, 9, 10, 12])
    @test string(bs) == "{2^[1 2 3 4]: ℕ(2)}"
    for i = 1:length(bs)
        @test searchsortedfirst(bs[i], bs) == i
    end
    @test Abelian(bs) == ℕ(2)

    bs = BinaryBases(1:2, 3:4, 𝕊ᶻ(1//2))
    @test collect(bs) == map(BinaryBasis{UInt}, [0b100, 0b1000, 0b1101, 0b1110])
    @test string(bs) == "{2^[1 2 3 4]: 𝕊ᶻ(1/2)}"
    @test Abelian(bs) == 𝕊ᶻ(1//2)

    bsdw = BinaryBases(1:2, ℕ(1)) ⊠ 𝕊ᶻ(-1//2)
    bsup = BinaryBases(3:4, ℕ(1)) ⊠ 𝕊ᶻ(1//2)
    bs = bsdw ⊗ bsup
    @test bs == BinaryBases(1:2, 3:4, ℕ(2) ⊠ 𝕊ᶻ(0))
    @test collect(bs) == map(BinaryBasis, [5, 6, 9, 10])
    @test string(bs) == "{2^[1 2]: ℕ(1) ⊠ 𝕊ᶻ(-1/2)} ⊗ {2^[3 4]: ℕ(1) ⊠ 𝕊ᶻ(1/2)}"
    @test Abelian(bs) ==  ℕ(2) ⊠ 𝕊ᶻ(0)

    bsdw = 𝕊ᶻ(-1//2) ⊠ BinaryBases(1:2, ℕ(1))
    bsup = 𝕊ᶻ(1//2) ⊠ BinaryBases(3:4, ℕ(1))
    bs = bsdw ⊗ bsup
    @test bs == BinaryBases(1:2, 3:4, 𝕊ᶻ(0) ⊠ ℕ(2))
    @test collect(bs) == map(BinaryBasis, [5, 6, 9, 10])
    @test string(bs) == "{2^[1 2]: 𝕊ᶻ(-1/2) ⊠ ℕ(1)} ⊗ {2^[3 4]: 𝕊ᶻ(1/2) ⊠ ℕ(1)}"
    @test Abelian(bs) ==  𝕊ᶻ(0) ⊠ ℕ(2)

    @test match(BinaryBases(2), BinaryBases(1:2))
    @test !match(BinaryBases(2), BinaryBases(1:2, ℕ(1)))

    @test !sumable(BinaryBases(1:2, ℕ(1)), BinaryBases(1:2))
    @test sumable(BinaryBases(1:2, ℕ(1)), BinaryBases(1:2, ℕ(2)))
    @test sumable(BinaryBases(1:2, ℕ(1)), BinaryBases(3:4, ℕ(1)))
    @test sumable(BinaryBases([ℕ(1)], [BinaryBasis(1:2)], [BinaryBasis(1)]), BinaryBases([ℕ(1)], [BinaryBasis(1:2)], [BinaryBasis(2)]))

    @test !productable(BinaryBases(1:2, ℕ(1)), BinaryBases(1:2))
    @test productable(BinaryBases(1:2, ℕ(1)), BinaryBases(3:4, ℕ(1)))
end

@testset "matrix" begin
    indexes = [𝕔(i, 1, 0, 1, [0.0, 0.0], [0.0, 0.0]) for i = 1:4]
    table = Table(indexes, OperatorIndexToTuple(:site, :orbital, :spin))

    braket = (BinaryBases(1:4, ℕ(2)), BinaryBases(1:4, ℕ(3)))
    ops = [Operator(2.0, index) for index in indexes]
    m₁ = SparseMatrixCSC(6, 4, [1, 2, 3, 4, 4], [3, 5, 6], [2.0, 2.0, 2.0])
    m₂ = SparseMatrixCSC(6, 4, [1, 2, 3, 3, 4], [2, 4, 6], [-2.0, -2.0, 2.0])
    m₃ = SparseMatrixCSC(6, 4, [1, 2, 2, 3, 4], [1, 4, 5], [2.0, -2.0, -2.0])
    m₄ = SparseMatrixCSC(6, 4, [1, 1, 2, 3, 4], [1, 2, 3], [2.0, 2.0, 2.0])
    @test matrix(ops[1], braket, table) == m₁
    @test matrix(ops[2], braket, table) == m₂
    @test matrix(ops[3], braket, table) == m₃
    @test matrix(ops[4], braket, table) == m₄
    @test matrix(ops[1]+ops[2]+ops[3]+ops[4], braket, table) == m₁+m₂+m₃+m₄

    braket = (BinaryBases(1:4, ℕ(2)), BinaryBases(1:4, ℕ(2)))
    @test matrix(ops[2]'*ops[1], braket, table) == SparseMatrixCSC(6, 6, [1, 1, 2, 2, 3, 3, 3], [3, 5], [4.0, 4.0])
    @test matrix(ops[3]'*ops[1], braket, table) == SparseMatrixCSC(6, 6, [1, 2, 2, 2, 3, 3, 3], [3, 6], [-4.0, 4.0])
end

@testset "EDMatrix & EDEigenData" begin
    m = EDMatrix(SparseMatrixCSC(6, 6, [1, 3, 6, 9, 11, 14, 16], [1, 2, 1, 3, 4, 2, 3, 5, 2, 5, 3, 4, 6, 5, 6], [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]), BinaryBases(1:4, ℕ(2)))
    @test m == EDMatrix(m.matrix, m.bra, m.ket) == EDMatrix(m.matrix, (m.bra, m.ket))
    @test parameternames(typeof(m)) == (:value, :sector)
    @test getcontent(m, :id) == (m.bra, m.ket)
    @test getcontent(m, :value) == m.matrix
    @test scalartype(m) == scalartype(typeof(m)) == Float64
    @test promote_type(typeof(m), ComplexF64) == EDMatrix{SparseMatrixCSC{ComplexF64, Int}, BinaryBases{ℕ, BinaryBasis{UInt}, Vector{BinaryBasis{UInt}}}}

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
    indexes = [𝕔(i, 1, 0, 1, [0.0, 0.0], [0.0, 0.0]) for i = 1:4]
    table = Table(indexes, OperatorIndexToTuple(:site, :orbital, :spin))
    op₁, op₂, op₃ = Operator(2.0, indexes[2]', indexes[1]), Operator(2.0, indexes[3]', indexes[2]), Operator(2.0, indexes[4]', indexes[3])
    ops = op₁ + op₂ + op₃
    target = TargetSpace([BinaryBases(1:4, ℕ(1)), BinaryBases(1:4, ℕ(2)), BinaryBases(1:4, ℕ(3))], table)
    M = EDMatrix{SparseMatrixCSC{Float64, Int}, BinaryBases{ℕ, BinaryBasis{UInt}, Vector{BinaryBasis{UInt}}}}

    mr = EDMatrixization{Float64}(target)
    @test valtype(typeof(mr), eltype(ops)) == valtype(typeof(mr), typeof(ops)) == OperatorSum{M, idtype(M)}

    ms = mr(ops)
    mr₁ = EDMatrixization{Float64}(target[1:1])
    mr₂ = EDMatrixization{Float64}(target[2:2])
    mr₃ = EDMatrixization{Float64}(target[3:3])
    @test ms == mr₁(ops) + mr₂(ops) + mr₃(ops)
    @test mr₁(ops) == mr₁(op₁) + mr₁(op₂) + mr₁(op₃)
    @test mr₂(ops) == mr₂(op₁) + mr₂(op₂) + mr₂(op₃)
    @test mr₃(ops) == mr₃(op₁) + mr₃(op₂) + mr₃(op₃)

    sf = SectorFilter(target[1])
    @test sf == SectorFilter((target[1], target[1]))
    @test valtype(typeof(sf), typeof(ms)) == typeof(ms)
    @test sf(ms) == mr₁(ops)
end

@testset "EDKind & Metric" begin
    fock = Fock{:f}(1, 2)
    hilbert = Hilbert(fock, 2)
    @test EDKind(hilbert) == EDKind(typeof(hilbert)) == EDKind(fock) == EDKind(typeof(fock)) == EDKind(:Binary)
    @test Metric(EDKind(hilbert), hilbert) == OperatorIndexToTuple(:spin, :site, :orbital)

    internalindex = 𝕔(1, 1, 1)
    index = 𝕔(1, 1, 1, 1)
    coordinatedindex = 𝕔(1, 1, 1, 1, [0.0], [0.0])
    @test EDKind(internalindex) == EDKind(typeof(internalindex)) == EDKind(index) == EDKind(typeof(index)) == EDKind(coordinatedindex) == EDKind(typeof(coordinatedindex)) == EDKind(:Binary)
end

@testset "Sector" begin
    hilbert = Hilbert(Fock{:f}(1, 2), 2)
    @test Sector(hilbert) == BinaryBases(4)
    @test Sector(hilbert, ℕ(2)) == BinaryBases(4, ℕ(2))
    @test Sector(hilbert, 𝕊ᶻ(1//2)) == BinaryBases(1:2, 3:4, 𝕊ᶻ(1//2))
    @test Sector(hilbert, ℕ(2) ⊠ 𝕊ᶻ(0)) == BinaryBases(1:2, 3:4, ℕ(2) ⊠ 𝕊ᶻ(0))
end

@testset "TargetSpace" begin
    hilbert = Hilbert(Fock{:f}(1, 2), 2)
    table = Table(hilbert, Metric(EDKind(hilbert), hilbert))

    bs₁ = BinaryBases(1:4, ℕ(2))
    bs₂ = BinaryBases(1:4, ℕ(3))
    ts = TargetSpace([bs₁, bs₂], table)
    @test getcontent(ts, :contents) == ts.sectors
    @test ts[2:2] == TargetSpace([bs₂], table)
    @test ts == add!(TargetSpace([bs₁], table), bs₂)
    @test ts == TargetSpace([bs₁], table)⊕bs₂ == bs₁⊕TargetSpace([bs₂], table)

    @test TargetSpace(hilbert) == TargetSpace([BinaryBases(4)], table)
    @test TargetSpace(hilbert, ℕ(2)) == TargetSpace(hilbert, ℕ(2), table) == TargetSpace([BinaryBases(4, ℕ(2))], table)
end

@testset "Binary ED" begin
    lattice = Lattice([0.0], [1.0])
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, 1.0, 1)
    U = Hubbard(:U, 0.0)
    μ = Onsite(:μ, 0.0)

    ed = Algorithm(Symbol("two-site"), ED(lattice, hilbert, (t, U, μ), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0); delay=true))
    @test kind(ed.frontend) == kind(typeof(ed.frontend)) == EDKind(:Binary)
    @test scalartype(ed.frontend) == scalartype(typeof(ed.frontend)) == Float64
    @test Parameters(ed) == (t=1.0, U=0.0, μ=0.0)

    vector = [0.5, -0.5, -0.5, 0.5]
    eigensystem = eigen(prepare!(ed); nev=1)
    values, vectors, sectors = eigensystem
    @test values==eigensystem.values && vectors==eigensystem.vectors && sectors==eigensystem.sectors
    @test isapprox(eigensystem.values[1], -2.0; atol=10^-10)
    @test isapprox(eigensystem.vectors[1], vector; atol=10^-10) || isapprox(eigensystem.vectors[1], -vector; atol=10^-10)
    eigensystem = eigen(matrix(ed); nev=1)
    @test isapprox(eigensystem.values[1], -2.0; atol=10^-10)
    @test isapprox(eigensystem.vectors[1], vector; atol=10^-10) || isapprox(eigensystem.vectors[1], -vector; atol=10^-10)

    vector = [-0.43516214649359913, 0.5573454101893041, 0.5573454101893037, -0.43516214649359913]
    update!(release!(ed); U=1.0, μ=-0.5)
    eigensystem = eigen(prepare!(ed), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0); nev=1)
    @test isapprox(eigensystem.values[1], -2.5615528128088303; atol=10^-10)
    @test isapprox(eigensystem.vectors[1], vector; atol=10^-10) || isapprox(eigensystem.vectors[1], -vector; atol=10^-10)

    vector = [-0.37174803446018434, 0.6015009550075453, 0.6015009550075459, -0.3717480344601846]
    update!(ed; U=2.0, μ=-1.0)
    eigensystem = eigen(ed, ℕ(length(lattice)) ⊠ 𝕊ᶻ(0); nev=1)
    @test isapprox(eigensystem.values[1], -3.23606797749979; atol=10^-10)
    @test isapprox(eigensystem.vectors[1], vector; atol=10^-10) || isapprox(eigensystem.vectors[1], -vector; atol=10^-10)

    another = ED(ed.frontend.system, TargetSpace(hilbert, ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)); delay=false)
    eigensystem = eigen(ed, ℕ(length(lattice)) ⊠ 𝕊ᶻ(0); nev=1)
    @test isapprox(eigensystem.values[1], -3.23606797749979; atol=10^-10)
    @test isapprox(eigensystem.vectors[1], vector; atol=10^-10) || isapprox(eigensystem.vectors[1], -vector; atol=10^-10)
end
