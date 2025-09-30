using ExactDiagonalization
using ExactDiagonalization: BinaryBasisRange, basistype
using QuantumLattices: Fock, Hilbert, Metric, Operator, OperatorIndexToTuple, Table
using QuantumLattices: 𝕔
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
    indexes = [𝕔(i, 1, 0, [0.0, 0.0], [0.0, 0.0]) for i = 1:4]
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
    @test matrix(Operator(2.5), braket, table) == SparseMatrixCSC(6, 6, [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6], [2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
    @test matrix(ops[2]'*ops[1], braket, table) == SparseMatrixCSC(6, 6, [1, 1, 2, 2, 3, 3, 3], [3, 5], [4.0, 4.0])
    @test matrix(ops[3]'*ops[1], braket, table) == SparseMatrixCSC(6, 6, [1, 2, 2, 2, 3, 3, 3], [3, 6], [-4.0, 4.0])
end

@testset "EDKind & Metric" begin
    fock = Fock{:f}(1, 2)
    hilbert = Hilbert(fock, 2)
    @test EDKind(hilbert) == EDKind(typeof(hilbert)) == EDKind(fock) == EDKind(typeof(fock)) == EDKind(:Binary)
    @test Metric(EDKind(hilbert), hilbert) == OperatorIndexToTuple(:spin, :site, :orbital)

    internalindex = 𝕔(1, 1)
    index = 𝕔(1, 1, 1)
    coordinatedindex = 𝕔(1, 1, 1, [0.0], [0.0])
    @test EDKind(internalindex) == EDKind(typeof(internalindex)) == EDKind(index) == EDKind(typeof(index)) == EDKind(coordinatedindex) == EDKind(typeof(coordinatedindex)) == EDKind(:Binary)
end

@testset "Sector" begin
    hilbert = Hilbert(Fock{:f}(1, 2), 2)
    @test Sector(hilbert) == BinaryBases(4)
    @test Sector(ℕ(2), hilbert) == BinaryBases(4, ℕ(2))
    @test Sector(𝕊ᶻ(1//2), hilbert) == BinaryBases(1:2, 3:4, 𝕊ᶻ(1//2))
    @test Sector(ℕ(2) ⊠ 𝕊ᶻ(0), hilbert) == BinaryBases(1:2, 3:4, ℕ(2) ⊠ 𝕊ᶻ(0))
    @test broadcast(Sector, (ℕ(2) ⊠ 𝕊ᶻ(0),), hilbert) == (BinaryBases(1:2, 3:4, ℕ(2) ⊠ 𝕊ᶻ(0)),)
end
