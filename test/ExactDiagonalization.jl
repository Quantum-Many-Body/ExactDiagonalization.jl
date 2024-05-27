using ExactDiagonalization
using ExactDiagonalization: BinaryBasisRange, sumable, productable
using LinearAlgebra: eigen
using QuantumLattices: ⊕, ⊗, add!, contentnames, dtype, getcontent, id, idtype, kind, matrix, parameternames, statistics, update!
using QuantumLattices: AbelianNumber, Algorithm, CompositeIndex, FID, Fock, Heisenberg, Hilbert, Hopping, Hubbard, Index, Lattice, Metric, Onsite, Operator, OperatorSum, OperatorUnitToTuple, Parameters, ParticleNumber, Spin, SpinfulParticle, Sz, Table
using SparseArrays: SparseMatrixCSC

@testset "BinaryBasis" begin
    @test basistype(Int8(1)) == UInt8
    @test basistype(Int16(1)) == UInt16
    @test basistype(Int32(1)) == UInt32
    @test basistype(Int64(1)) == UInt64
    @test basistype(Int128(1)) == UInt128

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
    bbr = BinaryBasisRange(2)
    @test issorted(bbr) == true
    @test length(bbr) == 4
    for i = 1:length(bbr)
        @test bbr[i] == BinaryBasis(UInt(i-1))
    end
end

@testset "BinaryBases" begin
    bs = BinaryBases(2)
    @test issorted(bs) == true
    @test length(bs) == length(bs) == 4
    @test bs == BinaryBases(1:2)
    @test isequal(bs, BinaryBases((2, 1)))
    @test hash(bs, UInt(1)) == hash(id(bs), UInt(1)) == hash((bs.quantumnumbers, bs.stategroups), UInt(1))
    for i = 1:length(bs)
        @test bs[i]==BinaryBasis(i-1)
        @test searchsortedfirst(bs[i], bs) == i
    end
    @test eltype(bs) == eltype(typeof(bs)) == BinaryBasis{UInt}
    @test collect(bs) == map(BinaryBasis, [0, 1, 2, 3])
    @test string(bs) == "{2^1:2}"

    bs = BinaryBases(4, 2)
    @test collect(bs) == map(BinaryBasis, [3, 5, 6, 9, 10, 12])
    @test string(bs) == "{2^[1 2 3 4]: ParticleNumber(2.0)}"
    for i = 1:length(bs)
        @test searchsortedfirst(bs[i], bs) == i
    end

    bs = BinaryBases{SpinfulParticle}(1:2, 1; Sz=-0.5) ⊗ BinaryBases{SpinfulParticle}(3:4, 1; Sz=0.5)
    @test collect(bs) == map(BinaryBasis, [5, 6, 9, 10])
    @test AbelianNumber(bs) == SpinfulParticle(2.0, 0.0)
    @test string(bs) == "{2^[1 2]: SpinfulParticle(1.0, -0.5)} ⊗ {2^[3 4]: SpinfulParticle(1.0, 0.5)}"
end

@testset "SpinBases" begin
    bs = SpinBases([1//2, 1//2])
    @test isequal(id(bs), (Sz(NaN), Rational{Int64}[1//2, 1//2], ([1], [2])))
    @test length(bs) == 4
    @test string(bs) == "{(1/2₁) ⊗ (1/2₂): Sz(NaN)}"
    @test match(bs, bs)
    @test isequal(AbelianNumber(bs), Sz(NaN))

    another = SpinBases([1//2, 1//2], Sz(0.0))
    @test id(another) == (Sz(0.0), Rational{Int64}[1//2, 1//2], ([1], [2]))
    @test length(another) == 2
    @test string(another) == "{(1/2₁) ⊗ (1/2₂): Sz(0.0)}"
    @test match(another, SpinBases([1//2, 1//2], Sz(1.0)))
    @test isequal(AbelianNumber(another), Sz(0.0))
    @test sumable(another, SpinBases([1//2, 1//2], Sz(1.0)))
end

@testset "TargetSpace" begin
    bs₁ = BinaryBases(1:4, 2)
    bs₂ = BinaryBases(1:4, 3)
    ts = TargetSpace(bs₁, bs₂)
    @test contentnames(typeof(ts)) == (:contents,)
    @test getcontent(ts, :contents) == ts.sectors
    @test add!(add!(TargetSpace(typeof(bs₁)[]), bs₁), bs₂) == add!(TargetSpace(typeof(bs₁)[]), ts) == ts
    @test bs₁⊕bs₂ == ts

    bs₃ = BinaryBases(5:8, 2)
    @test ts⊕bs₃ == TargetSpace(bs₁, bs₂, bs₃)
    @test bs₃⊕ts == TargetSpace(bs₃, bs₁, bs₂)
    @test ts⊕TargetSpace(bs₃) == TargetSpace(bs₁, bs₂, bs₃)
end

@testset "matrix" begin
    indexes = [CompositeIndex(Index(i, FID{:f}(1, 0, 1)), [0.0, 0.0], [0.0, 0.0]) for i = 1:4]
    table = Table(indexes, OperatorUnitToTuple(:site, :orbital, :spin))

    braket = (BinaryBases(1:4, 2), BinaryBases(1:4, 3))
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
    @test promote_type(typeof(m), ComplexF64) == EDMatrix{BinaryBases{ParticleNumber, BinaryBasis{UInt}, Vector{BinaryBasis{UInt}}}, SparseMatrixCSC{ComplexF64, Int}}
end

@testset "EDMatrixRepresentation && SectorFilter" begin
    indexes = [CompositeIndex(Index(i, FID{:f}(1, 0, 1)), [0.0, 0.0], [0.0, 0.0]) for i = 1:4]
    table = Table(indexes, OperatorUnitToTuple(:site, :orbital, :spin))
    op₁, op₂, op₃ = Operator(2.0, indexes[2]', indexes[1]), Operator(2.0, indexes[3]', indexes[2]), Operator(2.0, indexes[4]', indexes[3])
    ops = op₁ + op₂ + op₃
    target = BinaryBases(1:4, 1)⊕BinaryBases(1:4, 2)⊕BinaryBases(1:4, 3)
    M = EDMatrix{BinaryBases{ParticleNumber, BinaryBasis{UInt}, Vector{BinaryBasis{UInt}}}, SparseMatrixCSC{Float64, Int}}

    mr = EDMatrixRepresentation{Float64}(target, table)
    @test valtype(typeof(mr), eltype(ops)) == valtype(typeof(mr), typeof(ops)) == OperatorSum{M, idtype(M)}

    ms = mr(ops)
    mr₁ = EDMatrixRepresentation{Float64}(TargetSpace(target[1]), table)
    mr₂ = EDMatrixRepresentation{Float64}(TargetSpace(target[2]), table)
    mr₃ = EDMatrixRepresentation{Float64}(TargetSpace(target[3]), table)
    @test ms == mr₁(ops) + mr₂(ops) + mr₃(ops)
    @test mr₁(ops) == mr₁(op₁) + mr₁(op₂) + mr₁(op₃)
    @test mr₂(ops) == mr₂(op₁) + mr₂(op₂) + mr₂(op₃)
    @test mr₃(ops) == mr₃(op₁) + mr₃(op₂) + mr₃(op₃)

    sf = SectorFilter(target[1])
    @test sf == SectorFilter((target[1], target[1]))
    @test valtype(typeof(sf), typeof(ms)) == typeof(ms)
    @test sf(ms) == mr₁(ops)
end

@testset "FED" begin
    @test EDKind(Hilbert{<:Fock}) == EDKind(:FED)
    @test Metric(EDKind(:FED), Hilbert(1=>Fock{:f}(1, 2))) == OperatorUnitToTuple(:spin, :site, :orbital)

    lattice = Lattice([0.0], [1.0])
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))

    @test Sector(hilbert) == BinaryBases(2*length(lattice))
    @test Sector(hilbert, ParticleNumber(NaN)) == BinaryBases(1:2*length(lattice))
    @test Sector(hilbert, ParticleNumber(length(lattice))) == BinaryBases(2*length(lattice), length(lattice))
    @test Sector(hilbert, SpinfulParticle(NaN, NaN)) == BinaryBases{SpinfulParticle}(1:2*length(lattice))
    @test Sector(hilbert, SpinfulParticle(length(lattice), NaN)) == BinaryBases{SpinfulParticle}(2*length(lattice), length(lattice); Sz=NaN)
    @test Sector(hilbert, SpinfulParticle(NaN, 0.5)) == BinaryBases([SpinfulParticle(NaN, 0.5)], [BinaryBasis{UInt}(0b1111)], map(BinaryBasis{UInt}, [0b100, 0b1000, 0b1101, 0b1110]))
    @test Sector(hilbert, SpinfulParticle(length(lattice), 0.0)) == BinaryBases{SpinfulParticle}(1:length(lattice), length(lattice)÷2; Sz=-0.5)⊗BinaryBases{SpinfulParticle}(length(lattice)+1:2*length(lattice), length(lattice)÷2; Sz=+0.5)

    @test TargetSpace(hilbert) == TargetSpace(BinaryBases(2*length(lattice)))
    @test TargetSpace(hilbert, ParticleNumber(length(lattice))) == TargetSpace(BinaryBases(2*length(lattice), length(lattice)))

    t = Hopping(:t, 1.0, 1)
    U = Hubbard(:U, 0.0)
    μ = Onsite(:μ, 0.0)

    ed = Algorithm(Symbol("two-site"), ED(lattice, hilbert, (t, U, μ), SpinfulParticle(length(lattice), 0.0)))
    @test kind(ed.frontend) == kind(typeof(ed.frontend)) == EDKind(:FED)
    @test valtype(ed.frontend) == valtype(typeof(ed.frontend)) == Float64
    @test statistics(ed.frontend) == statistics(typeof(ed.frontend)) == :f
    @test Parameters(ed) == (t=1.0, U=0.0, μ=0.0)

    vector = [0.5, -0.5, -0.5, 0.5]
    eigensystem = eigen(ed; nev=1)
    values, vectors, sectors = eigensystem
    @test values==eigensystem.values && vectors==eigensystem.vectors && sectors==eigensystem.sectors
    @test isapprox(eigensystem.values[1], -2.0; atol=10^-10)
    @test isapprox(eigensystem.vectors[1], vector; atol=10^-10) || isapprox(eigensystem.vectors[1], -vector; atol=10^-10)

    vector = [-0.43516214649359913, 0.5573454101893041, 0.5573454101893037, -0.43516214649359913]
    update!(ed, U=1.0, μ=-0.5)
    eigensystem = eigen(ed, SpinfulParticle(length(lattice), 0.0); nev=1)
    @test isapprox(eigensystem.values[1], -2.5615528128088303; atol=10^-10)
    @test isapprox(eigensystem.vectors[1], vector; atol=10^-10) || isapprox(eigensystem.vectors[1], -vector; atol=10^-10)
end

@testset "SED" begin
    @test EDKind(Hilbert{<:Spin}) == EDKind(:SED)
    @test Metric(EDKind(:SED), Hilbert(1=>Spin{1//2}())) == OperatorUnitToTuple(:site)

    unitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (4, 4))
    hilbert = Hilbert(site=>Spin{1//2}() for site=1:length(lattice))

    ed = ED(lattice, hilbert, Heisenberg(:J, 1.0, 1), (Sz(0.0), Sz(1.0), Sz(-1.0)))
    eigensystem = eigen(matrix(ed); nev=4)
    @test isapprox(eigensystem.values, [-9.189207065192935, -8.686937479074416, -8.686937479074407, -8.686937479074404]; atol=10^-12)

    ed = ED(lattice, hilbert, Heisenberg(:J, 1.0, 1))
    eigensystem = eigen(matrix(ed); nev=6)
    @test isapprox(eigensystem.values[1:4], [-9.189207065192946, -8.686937479074421, -8.686937479074418, -8.68693747907441]; atol=10^-12)
end

@testset "spincoherentstates" begin
    unitcell = Lattice([0.0, 0.0], [0.0, √3/3]; vectors=[[1.0, 0.0], [0.5, √3/2]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))

    spins=Dict(i=>( isodd(i) ?  [0,0,1] : [0,0,-1] )    for i=1:length(lattice))
    state=spincoherentstates(xyz2ang(spins))
    @test findmax(state.|>abs) == (1.0,171)
    
    hilbert = Hilbert(Spin{1//2}(), length(lattice))
    targetspace = TargetSpace(hilbert)

    k,s=structure_factor(lattice,targetspace[1],hilbert,state)
    @test isapprox(s[3,2,11], 1.418439381905401)
    @test isapprox(structure_factor(lattice,targetspace[1],hilbert,state,[0.0,4*pi/sqrt(3)])[3],1.25)

    sp=Dict(1:2:length(lattice)|>collect=>[0.0,0.0],2:2:length(lattice)|>collect=>[pi,0.0])
    seta,p,pscs=Pspincoherentstates(state,sp)
    @test isapprox(pscs[:,1],[1.0 for _=1:length(pscs[:,1])])
end
