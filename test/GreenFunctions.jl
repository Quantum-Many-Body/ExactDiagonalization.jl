using ExactDiagonalization
using KrylovKit: ModifiedGramSchmidt, ModifiedGramSchmidt2
using LinearAlgebra: Diagonal, Hermitian, I, dot, inv, norm, tr
using QuantumLattices
using TightBindingApproximation
import CairoMakie as Makie
import Plots

@testset "GreenFunctionMethod" begin
    @test string(ExactMethod()) == "ExactMethod()"

    method = BandLanczosMethod()
    @test string(method) == "BandLanczosMethod(2.0e-8, 1.0e-8, true, 200, KrylovKit.ModifiedGramSchmidt2(), KrylovKit.ModifiedGramSchmidt())"
    @test method.tol == 2e-8
    @test method.tolᵣ == 1e-8
    @test method.keepvecs == true
    @test method.maxdim == 200
    @test method.orthᵣ == ModifiedGramSchmidt2()
    @test method.orthₒ == ModifiedGramSchmidt()

    method = BandLanczosMethod(tol=1e-5, tolᵣ=1e-12, keepvecs=false, maxdim=100)
    @test string(method) == "BandLanczosMethod(1.0e-5, 1.0e-12, false, 100, KrylovKit.ModifiedGramSchmidt2(), KrylovKit.ModifiedGramSchmidt())"
    @test method.tol == 1e-5
    @test method.tolᵣ == 1e-12
    @test method.keepvecs == false
    @test method.maxdim == 100

    method = BandLanczosMethod(ModifiedGramSchmidt(), ModifiedGramSchmidt2())
    @test string(method) == "BandLanczosMethod(2.0e-8, 1.0e-8, true, 200, KrylovKit.ModifiedGramSchmidt(), KrylovKit.ModifiedGramSchmidt2())"
    @test method.orthᵣ == ModifiedGramSchmidt()
    @test method.orthₒ == ModifiedGramSchmidt2()

    method = ThermalExactMethod(2.0)
    @test string(method) == "ThermalExactMethod(2.0, 0.0)"
    @test method.β == 2.0
    @test method.cutoff == 0.0

    method = ThermalExactMethod(2.0; cutoff=0.1)
    @test string(method) == "ThermalExactMethod(2.0, 0.1)"
    @test method.cutoff == 0.1
    @test_throws AssertionError ThermalExactMethod(-1.0)
    @test_throws AssertionError ThermalExactMethod(1.0; cutoff=1.5)

    method = ThermalSeedBandLanczosMethod(β=2.0)
    @test string(method) == "ThermalSeedBandLanczosMethod(2.0, 0.001, 10, 200, 200, 0, 2.0e-8, 1.0e-8, KrylovKit.ModifiedGramSchmidt2(), KrylovKit.ModifiedGramSchmidt())"
    @test method.β == 2.0
    @test method.cutoff == 1e-3
    @test method.nrandom == 10
    @test method.maxdim₁ == 200
    @test method.maxdim₂ == 200
    @test method.seed == 0
    @test method.tol == 2e-8
    @test method.tolᵣ == 1e-8
    @test method.orthᵣ == ModifiedGramSchmidt2()
    @test method.orthₒ == ModifiedGramSchmidt()
    @test isempty(method.samples)
    @test_throws AssertionError ThermalSeedBandLanczosMethod(β=0.0)
    @test_throws AssertionError ThermalSeedBandLanczosMethod(β=1.0, cutoff=1.5)
    @test_throws AssertionError ThermalSeedBandLanczosMethod(β=1.0, nrandom=0)
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
    g_exact = GreenFunction(operators, ed, GreenFunctionKind(:greater), ExactMethod())
    @test g_exact == GreenFunction(copy(g_exact.L), copy(g_exact.E))
    @test isequal(g_exact, GreenFunction(copy(g_exact.L), copy(g_exact.E)))
    @test g_exact == GreenFunction(copy(g_exact.L), copy(g_exact.R), copy(g_exact.E))
    @test isapprox(GreenFunction(copy(g_exact.L), copy(g_exact.E))(0.3+0.1im), g_exact(0.3+0.1im); atol=1e-12)
    @test rank(g_exact) == length(operators)
    @test count(g_exact) > 0
    @test eltype(g_exact) == Float64  # U=0 case gives real-valued Green function
    @test size(g_exact) == (length(operators), length(operators))
    @test size(g_exact(0.0 + 0.1im)) == (length(operators), length(operators))

    # Compute GreenFunction with three different methods:
    # 1. ExactMethod (full diagonalization)
    # 2. BandLanczosMethod with keepvecs=false
    # 3. BandLanczosMethod with keepvecs=true
    g_no_keep = GreenFunction(operators, ed, GreenFunctionKind(:greater), BandLanczosMethod(maxdim=200, keepvecs=false))
    g_keep = GreenFunction(operators, ed, GreenFunctionKind(:greater), BandLanczosMethod(maxdim=200, keepvecs=true))

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
    g_exact_lesser = GreenFunction(operators, ed, GreenFunctionKind(:lesser), ExactMethod())
    g_no_keep_lesser = GreenFunction(operators, ed, GreenFunctionKind(:lesser), BandLanczosMethod(maxdim=200, keepvecs=false))
    g_keep_lesser = GreenFunction(operators, ed, GreenFunctionKind(:lesser), BandLanczosMethod(maxdim=200, keepvecs=true))
    for ω in test_frequencies
        G_exact = g_exact_lesser(ω)
        G_no_keep = g_no_keep_lesser(ω)
        G_keep = g_keep_lesser(ω)
        @test isapprox(G_exact, G_no_keep; atol=1e-6)
        @test isapprox(G_exact, G_keep; atol=1e-6)
        @test isapprox(G_no_keep, G_keep; atol=1e-6)
    end
end

@testset "GreenFunctionKind(:retarded): comparison of all methods" begin
    # Test the retarded Green function with all methods using a 6-site lattice
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

    # Basic tests for the retarded Green function using ExactMethod
    rg_exact = GreenFunction(operators, ed, GreenFunctionKind(:retarded), ExactMethod(); sign=false)
    @test rank(rg_exact) == length(operators)

    # Compute the retarded Green function with three different methods:
    # 1. ExactMethod
    # 2. BandLanczosMethod with keepvecs=false
    # 3. BandLanczosMethod with keepvecs=true
    rg_no_keep = GreenFunction(operators, ed, GreenFunctionKind(:retarded), BandLanczosMethod(maxdim=200, keepvecs=false); sign=false)
    rg_keep = GreenFunction(operators, ed, GreenFunctionKind(:retarded), BandLanczosMethod(maxdim=200, keepvecs=true); sign=false)

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

@testset "GreenFunction ED vs TBA" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
    lattice = Lattice(unitcell, (6,))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, -1.0, 1)
    ed = Algorithm(Symbol("1D"), ED(lattice, hilbert, t, ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
    operators = mapreduce(vcat, hilbert) do pair
        site, fock = pair
        return [Index(site, fock[i]) for i=1:length(fock)÷2]
    end
    gf = GreenFunction(operators, ed, GreenFunctionKind(:retarded), BandLanczosMethod(); sign=false)
    tba = Algorithm(Symbol("1D"), TBA(lattice, hilbert, t))
    ω = rand(ComplexF64)
    @test isapprox(gf(ω), inv(ω*I-matrix(tba).H); atol=10^-10)
end

@testset "GreeFunction ED vs BdG" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
    lattice = Lattice(unitcell, (6,))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, -1.0, 1)
    Δ = Pairing(:Δ, rand(), 1, 𝕔𝕔(:, :, :); amplitude=bond->real(exp(im*azimuth(rcoordinate(bond)))))
    μ = Onsite(:μ, -1.2)
    quantumnumber = ℤ₁()
    ed = Algorithm(Symbol("1D"), ED(lattice, hilbert, (t, Δ, μ), quantumnumber))
    operators = sort!(mapreduce(vcat, hilbert) do pair
        site, fock = pair
        return [Index(site, fock[i]) for i=1:length(fock)]
    end;
    by=OperatorIndexToTuple(:nambu, :site, :orbital, :spin)
    )
    gf = GreenFunction(operators, ed, GreenFunctionKind(:retarded), BandLanczosMethod(); sign=false)
    tba = Algorithm(Symbol("1D"), TBA(lattice, hilbert, (t, Δ, μ)))
    ω = rand(ComplexF64)
    @test isapprox(gf(ω), inv(ω*I-matrix(tba).H); atol=10^-10)

    Δ = Pairing(:Δ, rand(), 0, 𝕔𝕔(:, :, real(1im*σʸ)))
    quantumnumber = 𝕊ᶻ(0)
    ed = Algorithm(Symbol("1D"), ED(lattice, hilbert, (t, Δ, μ), quantumnumber))
    gf = GreenFunction(operators, ed, GreenFunctionKind(:retarded), BandLanczosMethod(); sign=false)
    tba = Algorithm(Symbol("1D"), TBA(lattice, hilbert, (t, Δ, μ)))
    ω = rand(ComplexF64)
    @test isapprox(gf(ω), inv(ω*I-matrix(tba).H); atol=10^-10)
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
    g = GreenFunction(operators, ed, GreenFunctionKind(:retarded), BandLanczosMethod(; keepvecs=true))

    emin = -10.0
    emax = 10.0
    N = 501
    η = 0.1
    es = LinRange(emin, emax, N)

    dos = zeros(N)
    for (i, e) in enumerate(es)
        dos[i] = -2imag(tr(g(e+η*1im)))
    end
    Plots.savefig(Plots.plot(es, dos; minorticks=true, minorgrid=true), "Plots-Hubbard-1d-10-DOS.png")
    Makie.save("Makie-Hubbard-1d-10-DOS.png", Makie.lines(es, dos))

    path = ReciprocalPath(unitcell, line"X₂-X₁"; length=length(lattice))
    spectral = zeros(N, length(path))
    for (i, e) in enumerate(es)
        data = g(e+η*1im)
        for (m, opₘ) in enumerate(operators), (n, opₙ) in enumerate(operators)
            disp = lattice[opₘ.site] - lattice[opₙ.site]
            for (j, momentum) in enumerate(path)
                spectral[i, j] += -imag(data[m, n]*exp(1im*dot(disp, momentum)))
            end
        end
    end
    Plots.savefig(Plots.plot(path, es, log.(1 .+ spectral); minorticks=true, minorgrid=true), "Plots-Hubbard-1d-10-spectral.png")
    Makie.save("Makie-Hubbard-1d-10-spectral.png", Makie.plot(path, es, log.(1 .+ spectral)))
end

@testset "GreenFunction Heisenberg" begin
    unitcell = Lattice([0.0]; vectors=[[1.0]])
    lattice = Lattice(unitcell, (20,), ('P',))
    hilbert = Hilbert(Spin{1//2}(), length(lattice))
    J = Heisenberg(:J, 1.0, 1)
    ed = Algorithm(:ED, ED(lattice, hilbert, J, 𝕊ᶻ(0)))
    operators = [𝕊{1//2}(i, '+') for i in eachindex(lattice)]
    g = GreenFunction(operators, ed, GreenFunctionKind(:greater), BandLanczosMethod(; keepvecs=true))

    emin = 0.0
    emax = 4.0
    N = 401
    η = 0.1
    es = LinRange(emin, emax, N)
    path = ReciprocalPath(unitcell, line"Γ₁-Γ₂"; length=length(lattice))
    spectral = zeros(N, length(path))

    for (i, e) in enumerate(es)
        data = g(e+η*1im)
        for (m, opₘ) in enumerate(operators), (n, opₙ) in enumerate(operators)
            disp = lattice[opₘ.site] - lattice[opₙ.site]
            for (j, momentum) in enumerate(path)
                spectral[i, j] += -imag(data[m, n]*exp(1im*dot(disp, momentum)))
            end
        end
    end
    Plots.savefig(Plots.plot(path, es, log.(1 .+ spectral); minorticks=true, minorgrid=true), "Plots-Heisenberg-1d-20-spectral.png")
    Makie.save("Makie-Heisenberg-1d-20-spectral.png", Makie.plot(path, es, log.(1 .+ spectral)))
end

@testset "ThermalExactMethod: free fermion vs TBA" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
    lattice = Lattice(unitcell, (4,))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, -1.0, 1)
    ed = Algorithm(Symbol("1D"), ED(lattice, hilbert, t, ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
    operators = mapreduce(vcat, hilbert) do pair
        site, fock = pair
        return [Index(site, fock[i]) for i=1:length(fock)÷2]
    end
    tba = Algorithm(Symbol("1D"), TBA(lattice, hilbert, t))

    # the greater/lesser Green functions carry the canonical (fixed-particle-number) occupation factors of the free-fermion modes
    ε, U = eigen(Hermitian(matrix(tba).H))
    εs = unique(ε)   # the two spin blocks are identical
    for β in (0.5, 2.0, 20.0)
        # canonical occupations of a 4-mode block with two particles (the Sz=0 sector has two per spin flavor)
        occ = zeros(4)
        Z = 0.0
        for i in 1:4, j in i+1:4
            w = exp(-β*(εs[i]+εs[j]))
            Z += w
            occ[i] += w
            occ[j] += w
        end
        occ ./= Z
        n̄ = [occ[findfirst(==(e), εs)] for e in ε]
        g_greater = GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalExactMethod(β))
        g_lesser = GreenFunction(map(adjoint, operators), ed, GreenFunctionKind(:lesser), ThermalExactMethod(β))
        for ω in [0.0, 0.5, 1.0, -0.5, -1.0] .+ 0.05im
            @test isapprox(g_greater(ω), U*Diagonal((1 .- n̄)./(ω .- ε))*U'; atol=1e-8)
            @test isapprox(g_lesser(ω), U*Diagonal(n̄./(ω .- ε))*U'; atol=1e-8)
        end
    end

    # the retarded Green function of free fermions is temperature independent
    for β in (0.5, 2.0, 20.0)
        gf = GreenFunction(operators, ed, GreenFunctionKind(:retarded), ThermalExactMethod(β); sign=false)
        for ω in [0.0, 0.5, 1.0, -0.5, -1.0] .+ 0.05im
            @test isapprox(gf(ω), inv(ω*I-matrix(tba).H); atol=1e-8)
        end
    end
end

@testset "ThermalExactMethod: free fermion vs TBA (grand canonical)" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
    lattice = Lattice(unitcell, (4,))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, -1.0, 1)
    μ = Onsite(:μ, -0.7)   # explicit chemical potential
    ed = Algorithm(Symbol("1D"), ED(lattice, hilbert, (t, μ), ℤ₁()))   # full Hilbert space, grand canonical
    operators = mapreduce(vcat, hilbert) do pair
        site, fock = pair
        return [Index(site, fock[i]) for i=1:length(fock)÷2]
    end
    tba = Algorithm(Symbol("1D"), TBA(lattice, hilbert, (t, μ)))
    ε, U = eigen(Hermitian(matrix(tba).H))
    for β in (0.5, 2.0, 20.0)
        f = 1 ./ (1 .+ exp.(β .* ε))   # plain Fermi function (μ inside ε): exact for free fermions in the grand-canonical ensemble
        g_greater = GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalExactMethod(β))
        g_lesser = GreenFunction(map(adjoint, operators), ed, GreenFunctionKind(:lesser), ThermalExactMethod(β))
        for ω in [0.0, 0.5, 1.0, -0.5, -1.0] .+ 0.05im
            @test isapprox(g_greater(ω), U*Diagonal((1 .- f)./(ω .- ε))*U'; atol=1e-8)
            @test isapprox(g_lesser(ω), U*Diagonal(f./(ω .- ε))*U'; atol=1e-8)
        end
    end
end

@testset "ThermalExactMethod: cutoff behavior" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
    lattice = Lattice(unitcell, (4,))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 4.0)
    μ = Onsite(:μ, -U.value/2)
    ed = Algorithm(Symbol("1D"), ED(lattice, hilbert, (t, U, μ), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
    operators = mapreduce(vcat, hilbert) do pair
        site, fock = pair
        return [Index(site, fock[i]) for i=1:length(fock)÷2]
    end
    ωs = [0.0, 0.5, 1.0, -0.5, -1.0] .+ 0.05im

    # the Boltzmann-weight cutoff truncates high-energy eigenstates
    g_full = GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalExactMethod(2.0))
    g_cut = GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalExactMethod(2.0; cutoff=1e-2))
    @test count(g_cut) < count(g_full)
    for ω in ωs
        @test isapprox(g_cut(ω), g_full(ω); atol=0.05)
    end

    # T->0 is not degraded by the cutoff: only the ground state survives anyway
    gT = GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalExactMethod(200.0; cutoff=1e-2))
    g0 = GreenFunction(operators, ed, GreenFunctionKind(:greater), BandLanczosMethod(maxdim=500))
    for ω in ωs
        @test isapprox(gT(ω), g0(ω); atol=1e-6)
    end
end

@testset "ThermalSeedBandLanczosMethod vs ThermalExactMethod" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
    lattice = Lattice(unitcell, (4,))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 4.0)
    μ = Onsite(:μ, -U.value/2)
    ed = Algorithm(Symbol("1D"), ED(lattice, hilbert, (t, U, μ), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
    operators = mapreduce(vcat, hilbert) do pair
        site, fock = pair
        return [Index(site, fock[i]) for i=1:length(fock)÷2]
    end
    ωs = [0.0, 0.5, 1.0, -0.5, -1.0] .+ 0.05im

    # statistical agreement with the exact method over a temperature grid, with the deviation bounded by 5 times the standard error of the mean estimated from the individual samples
    for β in (0.5, 1.0, 2.0, 5.0, 20.0)
        method = ThermalSeedBandLanczosMethod(β=β, nrandom=50, seed=UInt64(12345), maxdim₁=500, maxdim₂=500)
        g_exact = GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalExactMethod(β))
        g_combined = GreenFunction(operators, ed, GreenFunctionKind(:greater), method)
        samples_g = copy(method.samples)
        l_exact = GreenFunction(operators, ed, GreenFunctionKind(:lesser), ThermalExactMethod(β))
        l_combined = GreenFunction(operators, ed, GreenFunctionKind(:lesser), method)
        samples_l = copy(method.samples)
        @test length(samples_g) == method.nrandom
        @test length(samples_l) == method.nrandom
        for ω in ωs, (exact, combined, samples) in ((g_exact, g_combined, samples_g), (l_exact, l_combined, samples_l))
            samplesω = [sample(ω) for sample in samples]
            meanω = sum(samplesω)/length(samplesω)
            σ = sqrt(sum(sampleω->sum(abs2, sampleω-meanω), samplesω)/(length(samplesω)-1)/length(samplesω))
            @test maximum(abs.(combined(ω)-exact(ω))) <= max(5σ, 0.05)
        end
    end

    # correct T->0 degradation: a single random vector suffices
    g0 = GreenFunction(operators, ed, GreenFunctionKind(:greater), BandLanczosMethod(maxdim=500))
    gT = GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalSeedBandLanczosMethod(β=200.0, nrandom=1, maxdim₁=500, maxdim₂=500))
    gE = GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalExactMethod(200.0))
    l0 = GreenFunction(operators, ed, GreenFunctionKind(:lesser), BandLanczosMethod(maxdim=500))
    lT = GreenFunction(operators, ed, GreenFunctionKind(:lesser), ThermalSeedBandLanczosMethod(β=200.0, nrandom=1, maxdim₁=500, maxdim₂=500))
    lE = GreenFunction(operators, ed, GreenFunctionKind(:lesser), ThermalExactMethod(200.0))
    for ω in ωs
        @test isapprox(gT(ω), g0(ω); atol=1e-6)
        @test isapprox(gE(ω), g0(ω); atol=1e-6)
        @test isapprox(lT(ω), l0(ω); atol=1e-6)
        @test isapprox(lE(ω), l0(ω); atol=1e-6)
    end

    # the sampling error shrinks with the number of random vectors
    β, ω = 2.0, 0.5+0.05im
    g_exact = GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalExactMethod(β))
    errs = map((20, 40, 80, 160)) do nrandom
        g = GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalSeedBandLanczosMethod(β=β, nrandom=nrandom, seed=UInt64(1010), maxdim₁=500, maxdim₂=500))
        return norm(g(ω)-g_exact(ω))
    end
    @test all(errs[i+1]<=errs[i] for i in 1:length(errs)-1)
    @test 1.5 <= errs[begin]/errs[end] <= 10

    # reproducibility of the random seed
    method = ThermalSeedBandLanczosMethod(β=2.0, nrandom=5, seed=UInt64(777), maxdim₁=500, maxdim₂=500)
    eigensystem = eigen(ed; nev=1)
    e₀, sector₀ = only(eigensystem.values), only(eigensystem.sectors)
    g₁ = GreenFunction(operators, ed, GreenFunctionKind(:greater), method; e₀=e₀, sector₀=sector₀)
    g₂ = GreenFunction(operators, ed, GreenFunctionKind(:greater), method; e₀=e₀, sector₀=sector₀)
    @test g₁ == g₂
end

@testset "GreenFunctionKind(:retarded): ThermalSeedBandLanczosMethod vs ThermalExactMethod" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
    lattice = Lattice(unitcell, (4,))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 4.0)
    μ = Onsite(:μ, -U.value/2)
    ed = Algorithm(Symbol("1D"), ED(lattice, hilbert, (t, U, μ), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
    operators = mapreduce(vcat, hilbert) do pair
        site, fock = pair
        return [Index(site, fock[i]) for i=1:length(fock)÷2]
    end
    ωs = [0.0, 0.5, 1.0, -0.5, -1.0] .+ 0.05im

    # statistical agreement of the retarded Green function with the exact method over a temperature grid, with the deviation bounded by 5 times the standard error of the mean estimated from the individual (per-sample merged) samples
    for β in (0.5, 2.0, 20.0)
        method = ThermalSeedBandLanczosMethod(β=β, nrandom=50, seed=UInt64(12345), maxdim₁=500, maxdim₂=500)
        r_exact = GreenFunction(operators, ed, GreenFunctionKind(:retarded), ThermalExactMethod(β))
        r_combined = GreenFunction(operators, ed, GreenFunctionKind(:retarded), method)
        samples = copy(method.samples)
        @test length(samples) == method.nrandom
        for ω in ωs
            samplesω = [sample(ω) for sample in samples]
            meanω = sum(samplesω)/length(samplesω)
            σ = sqrt(sum(sampleω->sum(abs2, sampleω-meanω), samplesω)/(length(samplesω)-1)/length(samplesω))
            @test maximum(abs.(r_combined(ω)-r_exact(ω))) <= max(5σ, 0.05)
        end
    end
end

@testset "ThermalSeedBandLanczosMethod vs ThermalExactMethod: Heisenberg chain" begin
    unitcell = Lattice([0.0]; vectors=[[1.0]])
    lattice = Lattice(unitcell, (8,), ('P',))
    hilbert = Hilbert(Spin{1//2}(), length(lattice))
    J = Heisenberg(:J, 1.0, 1)
    ed = Algorithm(:ED, ED(lattice, hilbert, J, 𝕊ᶻ(0)))
    operators = [𝕊{1//2}(i, '+') for i in eachindex(lattice)]
    ωs = [0.0, 0.5, 1.0, -0.5] .+ 0.05im

    # statistical agreement between the seed and exact methods on a spin chain (exercising the AbelianBases path)
    for β in (1.0, 5.0), kind in (GreenFunctionKind(:greater), GreenFunctionKind(:lesser))
        method = ThermalSeedBandLanczosMethod(β=β, nrandom=50, seed=UInt64(12345), maxdim₁=500, maxdim₂=500)
        g_exact = GreenFunction(operators, ed, kind, ThermalExactMethod(β))
        g_combined = GreenFunction(operators, ed, kind, method)
        samples = copy(method.samples)
        @test length(samples) == method.nrandom
        for ω in ωs
            samplesω = [sample(ω) for sample in samples]
            meanω = sum(samplesω)/length(samplesω)
            σ = sqrt(sum(sampleω->sum(abs2, sampleω-meanω), samplesω)/(length(samplesω)-1)/length(samplesω))
            @test maximum(abs.(g_combined(ω)-g_exact(ω))) <= max(5σ, 0.05)
        end
    end
end

@testset "GreenFunction sum rule: pole-weight completeness" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
    lattice = Lattice(unitcell, (4,))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 4.0)
    μ = Onsite(:μ, -U.value/2)
    ed = Algorithm(Symbol("1D"), ED(lattice, hilbert, (t, U, μ), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
    operators = mapreduce(vcat, hilbert) do pair
        site, fock = pair
        return [Index(site, fock[i]) for i=1:length(fock)÷2]
    end
    Id = Matrix{Float64}(I, length(operators), length(operators))
    weight(gf) = gf.L * transpose(gf.R)

    # the pole weights of the retarded Green function sum to the anticommutator identity, Σₖ L[i,k]R[j,k] = ⟨{Aᵢ, Aⱼ†}⟩ = δᵢⱼ
    for method in (ExactMethod(), BandLanczosMethod(maxdim=500))
        gf = GreenFunction(operators, ed, GreenFunctionKind(:retarded), method)
        @test isapprox(weight(gf), Id; atol=1e-8)
    end
    for β in (0.5, 2.0, 20.0)
        gf = GreenFunction(operators, ed, GreenFunctionKind(:retarded), ThermalExactMethod(β))
        @test isapprox(weight(gf), Id; atol=1e-10)
    end

    # the seed method obeys the identity within the statistical error of the individual samples
    method = ThermalSeedBandLanczosMethod(β=2.0, nrandom=50, seed=UInt64(12345), maxdim₁=500, maxdim₂=500)
    gf = GreenFunction(operators, ed, GreenFunctionKind(:retarded), method)
    samples = copy(method.samples)
    @test length(samples) == method.nrandom
    ws = [weight(sample) for sample in samples]
    meanw = sum(ws)/length(ws)
    σ = sqrt(sum(w->sum(abs2, w-meanw), ws)/(length(ws)-1)/length(ws))
    @test maximum(abs.(weight(gf)-Id)) <= max(5σ, 0.05)
end
