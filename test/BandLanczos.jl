using ExactDiagonalization.BandLanczos
using KrylovKit: Block, BlockLanczosIterator, initialize, expand!, rayleighquotient, basis, residual, rayleighextension
using LinearAlgebra: norm

@testset "BandLanczosFactorization & BandLanczosIterator" begin
    N = 16
    m = rand(ComplexF64, N, N)
    m = m+m'
    vs = [rand(ComplexF64, N), rand(ComplexF64, N), rand(ComplexF64, N)]

    iter₀ = BlockLanczosIterator(m, Block(vs), N)
    fact₀ = initialize(iter₀)
    iter₁ = BandLanczosIterator(m, Block(vs), N)
    fact₁ = initialize(iter₁)
    iter₂ = BandLanczosIterator(m, Block(vs), N; keepvecs=false)
    fact₂ = initialize(iter₂)

    niter = 5
    tol = 1e-13
    for i = 1:niter
        V = hcat(basis(fact₁).basis...)
        R = hcat(residual(fact₁).vec...)
        @test isapprox(norm(iter₁.operator*V-V*rayleighquotient(fact₁)-R*rayleighextension(fact₁)'), 0.0; atol=tol)
        @test isapprox(norm(fact₀.H[1:fact₀.k, 1:fact₀.k]-rayleighquotient(fact₁)), 0.0; atol=tol)
        @test isapprox(norm(rayleighquotient(fact₁)-rayleighquotient(fact₂)), 0.0; atol=tol)
        if i<niter
            expand!(iter₀, fact₀)
            expand!(iter₁, fact₁)
            expand!(iter₂, fact₂)
        end
    end
end
