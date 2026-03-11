using ExactDiagonalization
using LinearAlgebra: dot, tr
using Plots: plot, savefig
using QuantumLattices

@testset "GreenFunctionMethod" begin
    method = BandLanczosMethod()
    @test method.tol == 1e-10
    @test method.keepvecs == false
    @test method.maxdim == 200

    method = BandLanczosMethod(tol=1e-5, keepvecs=true, maxdim=100)
    @test method.tol == 1e-5
    @test method.keepvecs == true
    @test method.maxdim == 100
end

@testset "GreenFunction: comparison of all methods" begin
    # Create a 6-site lattice for testing all methods
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
    lattice = Lattice(unitcell, (6,))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 0.0)
    μ = Onsite(:μ, 0.0)
    ed = Algorithm(Symbol("1D"), ED(lattice, hilbert, (t, U, μ), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
    operators = mapreduce(vcat, hilbert) do pair
        site, fock = pair
        return [Index(site, fock[i]) for i=1:length(fock)÷2]
    end

    # Basic tests for GreenFunction using ExactMethod
    g_exact = GreenFunction(operators, ed, ExactMethod(); kind=:greater)
    @test g_exact == GreenFunction(copy(g_exact.Q), copy(g_exact.E))
    @test isequal(g_exact, GreenFunction(copy(g_exact.Q), copy(g_exact.E)))
    @test rank(g_exact) == length(operators)
    @test dimension(g_exact) > 0
    @test eltype(g_exact) == Float64  # U=0 case gives real-valued Green function
    @test size(g_exact) == (length(operators), length(operators))
    @test size(g_exact(0.0 + 0.1im)) == (length(operators), length(operators))

    # Compute GreenFunction with three different methods:
    # 1. ExactMethod (full diagonalization)
    # 2. BandLanczosMethod with keepvecs=false
    # 3. BandLanczosMethod with keepvecs=true
    g_no_keep = GreenFunction(operators, ed, BandLanczosMethod(maxdim=200, keepvecs=false); kind=:greater)
    g_keep = GreenFunction(operators, ed, BandLanczosMethod(maxdim=200, keepvecs=true); kind=:greater)

    # Evaluate at multiple frequency points and compare all methods
    test_frequencies = [0.0, 0.5, 1.0, -0.5, -1.0] .+ 0.05im
    for ω in test_frequencies
        G_exact = g_exact(ω)
        G_no_keep = g_no_keep(ω)
        G_keep = g_keep(ω)
        # All three methods should give the same result within numerical tolerance
        @test isapprox(G_exact, G_no_keep; atol=1e-6)
        @test isapprox(G_exact, G_keep; atol=1e-6)
        @test isapprox(G_no_keep, G_keep; atol=1e-6)
    end

    # Also test lesser Green function
    g_exact_lesser = GreenFunction(operators, ed, ExactMethod(); kind=:lesser)
    g_no_keep_lesser = GreenFunction(operators, ed, BandLanczosMethod(maxdim=200, keepvecs=false); kind=:lesser)
    g_keep_lesser = GreenFunction(operators, ed, BandLanczosMethod(maxdim=200, keepvecs=true); kind=:lesser)
    for ω in test_frequencies
        G_exact = g_exact_lesser(ω)
        G_no_keep = g_no_keep_lesser(ω)
        G_keep = g_keep_lesser(ω)
        @test isapprox(G_exact, G_no_keep; atol=1e-6)
        @test isapprox(G_exact, G_keep; atol=1e-6)
        @test isapprox(G_no_keep, G_keep; atol=1e-6)
    end
end

@testset "RetardedGreenFunction: comparison of all methods" begin
    # Test RetardedGreenFunction with all methods using a 6-site lattice
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
    lattice = Lattice(unitcell, (6,))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 0.0)
    μ = Onsite(:μ, 0.0)
    ed = Algorithm(Symbol("1D"), ED(lattice, hilbert, (t, U, μ), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
    operators = mapreduce(vcat, hilbert) do pair
        site, fock = pair
        return [Index(site, fock[i]) for i=1:length(fock)÷2]
    end

    # Basic tests for RetardedGreenFunction using ExactMethod
    rg_exact = RetardedGreenFunction(operators, ed, ExactMethod(); sign=false)
    @test rank(rg_exact) == length(operators)

    # Compute RetardedGreenFunction with three different methods:
    # 1. ExactMethod
    # 2. BandLanczosMethod with keepvecs=false
    # 3. BandLanczosMethod with keepvecs=true
    rg_no_keep = RetardedGreenFunction(operators, ed, BandLanczosMethod(maxdim=200, keepvecs=false); sign=false)
    rg_keep = RetardedGreenFunction(operators, ed, BandLanczosMethod(maxdim=200, keepvecs=true); sign=false)

    # Evaluate at multiple frequency points
    test_frequencies = [0.0, 0.5, 1.0, -0.5, -1.0] .+ 0.05im
    for ω in test_frequencies
        G_exact = rg_exact(ω)
        G_no_keep = rg_no_keep(ω)
        G_keep = rg_keep(ω)
        @test isapprox(G_exact, G_no_keep; atol=1e-6)
        @test isapprox(G_exact, G_keep; atol=1e-6)
        @test isapprox(G_no_keep, G_keep; atol=1e-6)
    end
end

@testset "GreenFunction Hubbard" begin
    unitcell = Lattice([0.0]; vectors=[[1.0]])
    lattice = Lattice(unitcell, (10,), ('P',))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 8.0)
    μ = Onsite(:μ, -U.value/2)
    ed = Algorithm(:ED, ED(lattice, hilbert, (t, U, μ), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
    operators = mapreduce(vcat, hilbert) do pair
        site, fock = pair
        return [Index(site, fock[i]) for i=1:length(fock)÷2]
    end
    g = RetardedGreenFunction(operators, ed, BandLanczosMethod(; keepvecs=true))

    emin = -10.0
    emax = 10.0
    N = 501
    η = 0.1
    es = LinRange(emin, emax, N)

    dos = zeros(N)
    for (i, e) in enumerate(es)
        dos[i] = -2imag(tr(g(e+η*1im)))
    end
    savefig(plot(es, dos; minorticks=true, minorgrid=true), "Hubbard-1d-10-DOS.png")

    path = ReciprocalPath(reciprocals(unitcell), line"X₂-X₁"; length=length(lattice))
    spectral = zeros(N, length(path))
    for (i, e) in enumerate(es)
        data = g(e+η*1im)
        for (m, opₘ) in enumerate(operators), (n, opₙ) in enumerate(operators)
            disp = lattice[opₘ.site] - lattice[opₙ.site]
            for (j, momentum) in enumerate(path)
                spectral[i, j] += -2*imag(data[m, n]*exp(1im*dot(disp, momentum)))
            end
        end
    end
    savefig(plot(path, es, log.(1 .+ spectral); minorticks=true, minorgrid=true), "Hubbard-1d-10-spectral.png")
end

@testset "GreenFunction Heisenberg" begin
    unitcell = Lattice([0.0]; vectors=[[1.0]])
    lattice = Lattice(unitcell, (20,), ('P',))
    hilbert = Hilbert(Spin{1//2}(), length(lattice))
    J = Heisenberg(:J, 1.0, 1)
    ed = Algorithm(:ED, ED(lattice, hilbert, J, 𝕊ᶻ(0)))
    operators = [𝕊{1//2}(i, '+') for i in eachindex(lattice)]
    g = GreenFunction(operators, ed, BandLanczosMethod(; keepvecs=true))

    emin = 0.0
    emax = 4.0
    N = 401
    η = 0.1
    es = LinRange(emin, emax, N)
    path = ReciprocalPath(reciprocals(unitcell), line"Γ₁-Γ₂"; length=length(lattice))
    spectral = zeros(N, length(path))

    for (i, e) in enumerate(es)
        data = g(e+η*1im)
        for (m, opₘ) in enumerate(operators), (n, opₙ) in enumerate(operators)
            disp = lattice[opₘ.site] - lattice[opₙ.site]
            for (j, momentum) in enumerate(path)
                spectral[i, j] += -2*imag(data[m, n]*exp(1im*dot(disp, momentum)))
            end
        end
    end
    savefig(plot(path, es, log.(1 .+ spectral); minorticks=true, minorgrid=true), "Heisenberg-1d-20-spectral.png")
end
