using ExactDiagonalization
using QuantumLattices: Hilbert, Metric, Operator, OperatorIndexToTuple, Spin, Table, 𝕊
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
    @test !sumable(bs, bs)
    @test Abelian(bs) == ℤ₁(0)
    @test range(bs) == 1:4

    locals = [Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1), Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)]
    another = AbelianBases(locals, 𝕊ᶻ(0))
    @test id(another) == (𝕊ᶻ(0), locals, ([1], [2]))
    @test dimension(another) == 2
    @test string(another) == "{[Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)₁] ⊗ [Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)₂]: 𝕊ᶻ(0)}"
    @test match(another, AbelianBases(locals, 𝕊ᶻ(1)))
    @test sumable(another, AbelianBases(locals, 𝕊ᶻ(1)))
    @test Abelian(another) == 𝕊ᶻ(0)
    @test range(another) == [2, 3]
end

@testset "matrix" begin
    indexes = [𝕊{1//2}(i, '+', [0.0, 0.0], [0.0, 0.0]) for i = 1:4]
    table = Table(indexes, OperatorIndexToTuple(:site))
    locals = fill(Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)', 4)

    ops = [Operator(2.0, index) for index in indexes]
    braket = (AbelianBases(locals, 𝕊ᶻ(1)), AbelianBases(locals, 𝕊ᶻ(0)))
    m₁ = SparseMatrixCSC(4, 6, [1, 2, 2, 2, 3, 4, 4], [1, 3, 4], [2.0, 2.0, 2.0])
    m₂ = SparseMatrixCSC(4, 6, [1, 2, 3, 4, 4, 4, 4], [2, 3, 4], [2.0, 2.0, 2.0])
    m₃ = SparseMatrixCSC(4, 6, [1, 1, 1, 2, 2, 3, 4], [1, 2, 3], [2.0, 2.0, 2.0])
    m₄ = SparseMatrixCSC(4, 6, [1, 1, 2, 2, 3, 3, 4], [1, 2, 4], [2.0, 2.0, 2.0])
    @test matrix(ops[1], braket, table) == m₁
    @test matrix(ops[2], braket, table) == m₂
    @test matrix(ops[3], braket, table) == m₃
    @test matrix(ops[4], braket, table) == m₄
    @test matrix(ops[1]+ops[2]+ops[3]+ops[4], braket, table) == m₁+m₂+m₃+m₄
    @test matrix(ops[1]', reverse(braket), table) == m₁'
    @test matrix(ops[2]', reverse(braket), table) == m₂'
    @test matrix(ops[3]', reverse(braket), table) == m₃'
    @test matrix(ops[4]', reverse(braket), table) == m₄'
    @test matrix((ops[1]+ops[2]+ops[3]+ops[4])', reverse(braket), table) == (m₁+m₂+m₃+m₄)'

    ops = [Operator(2.0, 𝕊{1//2}(i, '+'), 𝕊{1//2}(i+1, '-')) for i = 1:3]
    braket = (AbelianBases(locals, 𝕊ᶻ(0)), AbelianBases(locals, 𝕊ᶻ(0)))
    m₁ = SparseMatrixCSC(6, 6, [1, 1, 1, 1, 2, 3, 3], [2, 3], [2.0, 2.0])
    m₂ = SparseMatrixCSC(6, 6, [1, 2, 3, 3, 3, 3, 3], [5, 6], [2.0, 2.0])
    m₃ = SparseMatrixCSC(6, 6, [1, 1, 1, 2, 2, 3, 3], [2, 4], [2.0, 2.0])
    @test matrix(ops[1], braket, table) == m₁
    @test matrix(ops[2], braket, table) == m₂
    @test matrix(ops[3], braket, table) == m₃
    @test matrix(ops[1]+ops[2]+ops[3], braket, table) == m₁+m₂+m₃
    @test matrix(ops[1]', braket, table) == m₁'
    @test matrix(ops[2]', braket, table) == m₂'
    @test matrix(ops[3]', braket, table) == m₃'
    @test matrix((ops[1]+ops[2]+ops[3])', braket, table) == (m₁+m₂+m₃)'
end

@testset "matrix consistency" begin
    N = 10
    partition = ([1, 2], [3, 4, 5], [6, 7, 8, 9, 10])
    indexes = [𝕊{1//2}(i, '+', [0.0, 0.0], [0.0, 0.0]) for i = 1:N]
    table = Table(indexes, OperatorIndexToTuple(:site))
    locals₁ = fill(2, N)
    braket₁ = (AbelianBases(locals₁, partition), AbelianBases(locals₁, partition))

    locals₂ = fill(Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)', N)
    braket₂ = (AbelianBases(locals₂, 𝕊ᶻ(1), partition), AbelianBases(locals₂, 𝕊ᶻ(0), partition))
    slices = map(range, braket₂)
    for i = 1:N
        op = Operator(2.0, indexes[i])
        m₁ = matrix(op, braket₁, table)[slices...]
        m₂ = matrix(op, braket₂, table)
        @test m₁ == m₂
    end

    braket₂ = (AbelianBases(locals₂, 𝕊ᶻ(0), partition), AbelianBases(locals₂, 𝕊ᶻ(0), partition))
    slices = map(range, braket₂)
    for i = 1:N
        op = Operator(2.0, 𝕊{1//2}(i, '+'), 𝕊{1//2}(i==N ? 1 : i+1, '-'))
        m₁ = matrix(op, braket₁, table)[slices...]
        m₂ = matrix(op, braket₂, table)
        @test m₁ == m₂
    end
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
    @test Sector(𝕊ᶻ(0), hilbert) == AbelianBases([Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)', Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)'], 𝕊ᶻ(0))
    @test broadcast(Sector, (𝕊ᶻ(0),), hilbert) == (AbelianBases([Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)', Graded{𝕊ᶻ}(-1/2=>1, 1/2=>1)'], 𝕊ᶻ(0)),)
end
