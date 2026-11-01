using ExactDiagonalization
using ExactDiagonalization: derivative, derivative!
using KrylovKit: ModifiedGramSchmidt, ModifiedGramSchmidt2
using LinearAlgebra: Diagonal, Hermitian, I, dot, inv, norm, tr
using QuantumLattices
using TightBindingApproximation
import CairoMakie as Makie
import Plots

@testset "GreenFunctionMethod" begin
    method = ExactMethod()
    @test string(method) == "ExactMethod()"
    @test contenttoconfig(method) === method
    @test ntasks(method) == 1

    method = BandLanczosMethod()
    @test string(method) == "BandLanczosMethod(2.0e-8, 1.0e-8, true, 200, KrylovKit.ModifiedGramSchmidt2(), KrylovKit.ModifiedGramSchmidt(), 1)"
    @test method.tol == 2e-8
    @test method.tolᵣ == 1e-8
    @test method.keepvecs == true
    @test method.maxdim == 200
    @test method.orthᵣ == ModifiedGramSchmidt2()
    @test method.orthₒ == ModifiedGramSchmidt()
    @test ntasks(method) == method.ntasks == 1
    @test contenttoconfig(method) == (method.tol, method.tolᵣ, method.keepvecs, method.maxdim, method.orthᵣ, method.orthₒ)
    @test_throws AssertionError BandLanczosMethod(ntasks=0)
    @test_throws AssertionError ThermalExactMethod(β=-1.0)
    @test_throws AssertionError ThermalExactMethod(β=1.0, cutoff=1.5)

    method = BandLanczosMethod(ModifiedGramSchmidt(), ModifiedGramSchmidt2(); tol=1e-5, tolᵣ=1e-12, keepvecs=false, maxdim=100)
    @test string(method) == "BandLanczosMethod(1.0e-5, 1.0e-12, false, 100, KrylovKit.ModifiedGramSchmidt(), KrylovKit.ModifiedGramSchmidt2(), 1)"
    @test method.tol == 1e-5
    @test method.tolᵣ == 1e-12
    @test method.keepvecs == false
    @test method.maxdim == 100
    @test method.orthᵣ == ModifiedGramSchmidt()
    @test method.orthₒ == ModifiedGramSchmidt2()

    method = ThermalExactMethod(β=2.0)
    @test string(method) == "ThermalExactMethod(2.0, 0.0)"
    @test method.β == 2.0
    @test method.cutoff == 0.0
    @test contenttoconfig(method) === method
    @test ntasks(method) == 1

    method = ThermalExactMethod(β=2.0, cutoff=0.1)
    @test string(method) == "ThermalExactMethod(2.0, 0.1)"
    @test method.cutoff == 0.1

    method = ThermalSeedBandLanczosMethod(β=2.0)
    @test string(method) == "ThermalSeedBandLanczosMethod(2.0, 0.001, 10, 200, 200, 0, 2.0e-8, 1.0e-8, KrylovKit.ModifiedGramSchmidt2(), KrylovKit.ModifiedGramSchmidt(), 1)"
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
    @test ntasks(method) == method.ntasks == 1
    @test isempty(method.samples)
    @test contenttoconfig(method) == (method.β, method.cutoff, method.nrandom, method.maxdim₁, method.maxdim₂, method.seed, method.tol, method.tolᵣ, method.orthᵣ, method.orthₒ)
    @test_throws AssertionError ThermalSeedBandLanczosMethod(β=0.0)
    @test_throws AssertionError ThermalSeedBandLanczosMethod(β=1.0, cutoff=1.5)
    @test_throws AssertionError ThermalSeedBandLanczosMethod(β=1.0, nrandom=0)
    @test_throws AssertionError ThermalSeedBandLanczosMethod(β=1.0, ntasks=0)
end

@testset "GreenFunction evaluation" begin
    Lreal = [1.0 0.2 0.1 0.3; 0.1 1.2 0.2 0.1; 0.2 0.1 1.1 0.2]
    Rreal = [1.1 0.1 0.2 0.1; 0.2 1.0 0.1 0.3; 0.1 0.2 1.3 0.1]
    E = [-1.0, -0.2, 0.3, 1.4]
    for (L, R) in ((Lreal, Rreal), (complex.(Lreal, 0.1.*Rreal), complex.(Rreal, -0.2.*Lreal)))
        gf = GreenFunction(L, R, E)
        for ω in (0.1, 0.1+0.05im)
            expected = zeros(promote_type(eltype(gf), typeof(ω)), size(gf))
            for i in 1:rank(gf), j in 1:rank(gf), p in 1:count(gf)
                expected[i, j] += gf.L[i, p]*gf.R[j, p]/(ω-gf.E[p])
            end
            @test isapprox(gf(ω), expected; atol=1e-11)
            initial = fill(eltype(expected)(0.125), size(gf))
            dest = copy(initial)
            @test gf(dest, ω) === dest
            @test isapprox(dest, expected; atol=1e-11)
            T = promote_type(eltype(gf), typeof(ω))
            workspace = Matrix{T}(undef, size(L))
            dest_workspace = copy(initial)
            @test gf(dest_workspace, ω; workspace=workspace) === dest_workspace
            @test isapprox(dest_workspace, expected; atol=1e-11)
            @test isapprox(gf(ω; workspace=workspace), expected; atol=1e-11)

            expected_inverse = inv(expected)
            @test isapprox(inv(gf, ω), expected_inverse; atol=1e-11)
            @test isapprox(inv(gf, ω; workspace=workspace), expected_inverse; atol=1e-11)
            dest_inverse = fill(eltype(expected)(2), size(gf))
            @test inv!(dest_inverse, gf, ω; workspace=workspace) === dest_inverse
            @test isapprox(dest_inverse, expected_inverse; atol=1e-11)

            expected_derivative = zeros(promote_type(eltype(gf), typeof(ω)), size(gf))
            for i in 1:rank(gf), j in 1:rank(gf), p in 1:count(gf)
                expected_derivative[i, j] -= gf.L[i, p]*gf.R[j, p]/(ω-gf.E[p])^2
            end
            @test isapprox(derivative(gf, ω), expected_derivative; atol=1e-11)
            @test isapprox(derivative(gf, ω; workspace=workspace), expected_derivative; atol=1e-11)
            dest_derivative = fill(eltype(expected)(2), size(gf))
            @test derivative!(dest_derivative, gf, ω; workspace=workspace) === dest_derivative
            @test isapprox(dest_derivative, expected_derivative; atol=1e-11)

            @test_throws AssertionError gf(copy(initial), ω; workspace=zeros(T, 1, 1))
        end
    end
end

@testset "GreenFunction: zero-temperature methods and TBA" begin
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
    eigensystem = eigen(ed; nev=1)
    initial = (e₀=only(eigensystem.values), v₀=only(eigensystem.vectors), sector₀=only(eigensystem.sectors))

    # Keep one automatic ground-state initialization; all other method comparisons reuse the same eigensystem.
    greater_info = nothing
    exact_greater = GreenFunction(operators, ed, GreenFunctionKind(:greater), ExactMethod(); callback=info->(greater_info=info))
    @test greater_info.Z == 1
    @test greater_info.Ω ≈ initial.e₀
    @test Abelian(greater_info.sector₀) == Abelian(initial.sector₀)
    @test exact_greater == GreenFunction(copy(exact_greater.L), copy(exact_greater.E))
    @test isequal(exact_greater, GreenFunction(copy(exact_greater.L), copy(exact_greater.E)))
    @test exact_greater == GreenFunction(copy(exact_greater.L), copy(exact_greater.R), copy(exact_greater.E))
    @test isapprox(GreenFunction(copy(exact_greater.L), copy(exact_greater.E))(0.3+0.1im), exact_greater(0.3+0.1im); atol=1e-12)
    @test rank(exact_greater) == length(operators)
    @test count(exact_greater) > 0
    @test eltype(exact_greater) == Float64
    @test size(exact_greater) == (length(operators), length(operators))
    @test size(exact_greater(0.0 + 0.1im)) == (length(operators), length(operators))

    lesser_info = nothing
    exact_lesser = GreenFunction(operators, ed, GreenFunctionKind(:lesser), ExactMethod(); initial..., callback=info->(lesser_info=info))
    @test lesser_info.Z == 1
    @test lesser_info.Ω ≈ initial.e₀
    @test Abelian(lesser_info.sector₀) == Abelian(initial.sector₀)
    test_frequencies = [0.0, 0.5, 1.0, -0.5, -1.0] .+ 0.05im
    for (kind, exact) in ((GreenFunctionKind(:greater), exact_greater), (GreenFunctionKind(:lesser), exact_lesser))
        no_keep = GreenFunction(operators, ed, kind, BandLanczosMethod(maxdim=200, keepvecs=false); initial...)
        keep = GreenFunction(operators, ed, kind, BandLanczosMethod(maxdim=200, keepvecs=true); initial...)
        for ω in test_frequencies
            G_exact, G_no_keep, G_keep = exact(ω), no_keep(ω), keep(ω)
            @test isapprox(G_exact, G_no_keep; atol=1e-6)
            @test isapprox(G_exact, G_keep; atol=1e-6)
            @test isapprox(G_no_keep, G_keep; atol=1e-6)
        end
    end

    retarded_info = nothing
    exact_retarded = GreenFunction(operators, ed, GreenFunctionKind(:retarded), ExactMethod(); sign=false, initial..., callback=info->(retarded_info=info))
    @test retarded_info.Z == 1
    @test retarded_info.Ω ≈ initial.e₀
    @test Abelian(retarded_info.sector₀) == Abelian(initial.sector₀)
    no_keep = GreenFunction(operators, ed, GreenFunctionKind(:retarded), BandLanczosMethod(maxdim=200, keepvecs=false); sign=false, initial...)
    band_info = nothing
    keep = GreenFunction(operators, ed, GreenFunctionKind(:retarded), BandLanczosMethod(maxdim=200, keepvecs=true); sign=false, initial..., callback=info->(band_info=info))
    @test band_info.Z == 1
    @test band_info.Ω ≈ initial.e₀
    @test Abelian(band_info.sector₀) == Abelian(initial.sector₀)
    @test rank(exact_retarded) == length(operators)
    for ω in test_frequencies
        G_exact, G_no_keep, G_keep = exact_retarded(ω), no_keep(ω), keep(ω)
        @test isapprox(G_exact, G_no_keep; atol=1e-6)
        @test isapprox(G_exact, G_keep; atol=1e-6)
        @test isapprox(G_no_keep, G_keep; atol=1e-6)
    end

    tba = Algorithm(Symbol("1D"), TBA(lattice, hilbert, t))
    ω = rand(ComplexF64)
    @test isapprox(keep(ω), inv(ω*I-matrix(tba).H); atol=10^-10)
end

@testset "GreenFunction: zero-temperature methods and BdG" begin
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

@testset "ThermalExactMethod: canonical free fermions" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
    lattice = Lattice(unitcell, (4,))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    qn = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    t = Hopping(:t, -1.0, 1)
    ed = Algorithm(Symbol("1D"), ED(lattice, hilbert, t, qn))
    operators = mapreduce(vcat, hilbert) do pair
        site, fock = pair
        return [Index(site, fock[i]) for i=1:length(fock)÷2]
    end
    eigensystem = eigen(ed; nev=1)
    e₀, v₀, sector₀ = only(eigensystem.values), only(eigensystem.vectors), only(eigensystem.sectors)
    E₀, U₀ = eigen(Hermitian(collect(sparse(ed.frontend, sector₀))))
    thermal = (E₀=E₀, U₀=U₀, sector₀=sector₀)
    tba = Algorithm(Symbol("1D"), TBA(lattice, hilbert, t))

    expected_Z = sum(exp.(-2.0.*(E₀.-e₀)))
    expected_Ω = e₀-log(expected_Z)/2.0
    thermal_info = nothing
    # the retarded Green function of free fermions is temperature independent
    for β in (0.5, 2.0, 20.0)
        gf = GreenFunction(operators, ed, GreenFunctionKind(:retarded), ThermalExactMethod(β=β); sign=false, thermal..., callback=β==2.0 ? (info->(thermal_info=info)) : nothing)
        for ω in [0.0, 0.5, 1.0, -0.5, -1.0] .+ 0.05im
            @test isapprox(gf(ω), inv(ω*I-matrix(tba).H); atol=1e-8)
        end
    end
    @test thermal_info.Z ≈ expected_Z
    @test thermal_info.Ω ≈ expected_Ω
    @test Abelian(thermal_info.sector₀) == Abelian(sector₀)
end

@testset "ThermalExactMethod: grand canonical free fermions" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
    lattice = Lattice(unitcell, (4,))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, -1.0, 1)
    μ = Onsite(:μ, -0.7)
    ed = Algorithm(Symbol("1D"), ED(lattice, hilbert, (t, μ), ℤ₁()))
    operators = mapreduce(vcat, hilbert) do pair
        site, fock = pair
        return [Index(site, fock[i]) for i=1:length(fock)÷2]
    end
    tba = Algorithm(Symbol("1D"), TBA(lattice, hilbert, (t, μ)))
    ε, U = eigen(Hermitian(matrix(tba).H))
    sector₀ = only(eigen(ed; nev=1).sectors)
    E₀, U₀ = eigen(Hermitian(collect(sparse(ed.frontend, sector₀))))
    e₀ = first(E₀)
    expected_Z = sum(exp.(-2.0.*(E₀.-e₀)))
    expected_Ω = e₀-log(expected_Z)/2.0
    thermal = (E₀=E₀, U₀=U₀, sector₀=sector₀)
    greater_info, lesser_info = nothing, nothing
    for β in (0.5, 2.0, 20.0)
        f = 1 ./ (1 .+ exp.(β .* ε))
        g_greater = GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalExactMethod(β=β); thermal..., callback=β==2.0 ? (info->(greater_info=info)) : nothing)
        g_lesser = GreenFunction(map(adjoint, operators), ed, GreenFunctionKind(:lesser), ThermalExactMethod(β=β); thermal..., callback=β==2.0 ? (info->(lesser_info=info)) : nothing)
        for ω in [0.0, 0.5, 1.0, -0.5, -1.0] .+ 0.05im
            @test isapprox(g_greater(ω), U*Diagonal((1 .- f)./(ω .- ε))*U'; atol=1e-8)
            @test isapprox(g_lesser(ω), U*Diagonal(f./(ω .- ε))*U'; atol=1e-8)
        end
    end
    for info in (greater_info, lesser_info)
        @test info.Z ≈ expected_Z
        @test info.Ω ≈ expected_Ω
        @test Abelian(info.sector₀) == Abelian(sector₀)
    end
end

@testset "GreenFunction: four-site Hubbard chain" begin
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
    eigensystem = eigen(ed; nev=1)
    e₀, v₀, sector₀ = only(eigensystem.values), only(eigensystem.vectors), only(eigensystem.sectors)
    initial = (e₀=e₀, v₀=v₀, sector₀=sector₀)
    E₀, U₀ = eigen(Hermitian(collect(sparse(ed.frontend, sector₀))))
    thermal = (E₀=E₀, U₀=U₀, sector₀=sector₀)

    # Retain automatic ThermalExact initialization once and reuse its result as the β=2 reference.
    g_full = GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalExactMethod(β=2.0))
    exact = (
        greater=Dict(β => β==2.0 ? g_full : GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalExactMethod(β=β); thermal...) for β in (0.5, 1.0, 2.0, 5.0, 20.0, 200.0)),
        lesser=Dict(β => GreenFunction(operators, ed, GreenFunctionKind(:lesser), ThermalExactMethod(β=β); thermal...) for β in (0.5, 1.0, 2.0, 5.0, 20.0, 200.0)),
        retarded=Dict(β => GreenFunction(operators, ed, GreenFunctionKind(:retarded), ThermalExactMethod(β=β); thermal...) for β in (0.5, 2.0, 20.0)),
    )
    g0 = GreenFunction(operators, ed, GreenFunctionKind(:greater), BandLanczosMethod(maxdim=500); initial...)
    l0 = GreenFunction(operators, ed, GreenFunctionKind(:lesser), BandLanczosMethod(maxdim=500); initial...)
    r_exact0 = GreenFunction(operators, ed, GreenFunctionKind(:retarded), ExactMethod(); initial...)
    r_lanczos0 = GreenFunction(operators, ed, GreenFunctionKind(:retarded), BandLanczosMethod(maxdim=500); initial...)

    @testset "ThermalExactMethod: cutoff behavior" begin
        g_cut = GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalExactMethod(β=2.0; cutoff=1e-2); thermal...)
        @test count(g_cut) < count(g_full)
        for ω in ωs
            @test isapprox(g_cut(ω), g_full(ω); atol=0.05)
        end

        # T->0 is not degraded by the cutoff: only the ground state survives anyway
        gT = GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalExactMethod(β=200.0; cutoff=1e-2); thermal...)
        for ω in ωs
            @test isapprox(gT(ω), g0(ω); atol=1e-6)
        end
    end

    @testset "ThermalSeedBandLanczosMethod vs ThermalExactMethod" begin
        greater_info, lesser_info = nothing, nothing
        # statistical agreement with the exact method over a temperature grid, with the deviation bounded by 5 times the standard error of the mean estimated from the individual samples
        for β in (0.5, 1.0, 2.0, 5.0, 20.0)
            method_g = ThermalSeedBandLanczosMethod(β=β, nrandom=50, seed=UInt64(12345), maxdim₁=500, maxdim₂=500)
            method_l = ThermalSeedBandLanczosMethod(β=β, nrandom=50, seed=UInt64(12345), maxdim₁=500, maxdim₂=500)
            g_combined = GreenFunction(operators, ed, GreenFunctionKind(:greater), method_g; e₀=e₀, sector₀=sector₀, callback=β==2.0 ? (info->(greater_info=info)) : nothing)
            l_combined = GreenFunction(operators, ed, GreenFunctionKind(:lesser), method_l; e₀=e₀, sector₀=sector₀, callback=β==2.0 ? (info->(lesser_info=info)) : nothing)
            samples_g, samples_l = copy(method_g.samples), copy(method_l.samples)
            @test length(samples_g) == method_g.nrandom
            @test length(samples_l) == method_l.nrandom
            for ω in ωs, (reference, combined, samples) in ((exact.greater[β], g_combined, samples_g), (exact.lesser[β], l_combined, samples_l))
                samplesω = [sample(ω) for sample in samples]
                meanω = sum(samplesω)/length(samplesω)
                σ = sqrt(sum(sampleω->sum(abs2, sampleω-meanω), samplesω)/(length(samplesω)-1)/length(samplesω))
                @test maximum(abs.(combined(ω)-reference(ω))) <= max(5σ, 0.05)
            end
        end
        @test greater_info.Z == lesser_info.Z
        @test greater_info.Ω == lesser_info.Ω
        for info in (greater_info, lesser_info)
            @test info.Z > 0
            @test info.Ω ≈ e₀-log(info.Z)/2.0
            @test Abelian(info.sector₀) == Abelian(sector₀)
        end

        # correct T->0 degradation: a single random vector suffices
        gT = GreenFunction(operators, ed, GreenFunctionKind(:greater), ThermalSeedBandLanczosMethod(β=200.0, nrandom=1, maxdim₁=500, maxdim₂=500); e₀=e₀, sector₀=sector₀)
        lT = GreenFunction(operators, ed, GreenFunctionKind(:lesser), ThermalSeedBandLanczosMethod(β=200.0, nrandom=1, maxdim₁=500, maxdim₂=500); e₀=e₀, sector₀=sector₀)
        for ω in ωs
            @test isapprox(gT(ω), g0(ω); atol=1e-6)
            @test isapprox(exact.greater[200.0](ω), g0(ω); atol=1e-6)
            @test isapprox(lT(ω), l0(ω); atol=1e-6)
            @test isapprox(exact.lesser[200.0](ω), l0(ω); atol=1e-6)
        end

        # the sampling error shrinks with the number of random vectors
        β, ω = 2.0, 0.5+0.05im
        errs = map((20, 40, 80, 160)) do nrandom
            method = ThermalSeedBandLanczosMethod(β=β, nrandom=nrandom, seed=UInt64(123), maxdim₁=500, maxdim₂=500)
            g = GreenFunction(operators, ed, GreenFunctionKind(:greater), method; e₀=e₀, sector₀=sector₀)
            return norm(g(ω)-exact.greater[β](ω))
        end
        @test all(errs[i+1]<=errs[i] for i in 1:length(errs)-1)
        @test 1.5 <= errs[begin]/errs[end] <= 10

        # reproducibility of the random seed with independent mutable method instances
        method₁ = ThermalSeedBandLanczosMethod(β=2.0, nrandom=5, seed=UInt64(777), maxdim₁=500, maxdim₂=500)
        method₂ = ThermalSeedBandLanczosMethod(β=2.0, nrandom=5, seed=UInt64(777), maxdim₁=500, maxdim₂=500)
        g₁ = GreenFunction(operators, ed, GreenFunctionKind(:greater), method₁; e₀=e₀, sector₀=sector₀)
        g₂ = GreenFunction(operators, ed, GreenFunctionKind(:greater), method₂; e₀=e₀, sector₀=sector₀)
        @test g₁ == g₂
    end

    @testset "GreenFunction sum rule: pole-weight completeness" begin
        Id = Matrix{Float64}(I, length(operators), length(operators))
        weight(gf) = gf.L * transpose(gf.R)

        # the pole weights of the retarded Green function sum to the anticommutator identity, Σₖ L[i,k]R[j,k] = ⟨{Aᵢ, Aⱼ†}⟩ = δᵢⱼ
        for gf in (r_exact0, r_lanczos0)
            @test isapprox(weight(gf), Id; atol=1e-8)
        end
        for β in (0.5, 2.0, 20.0)
            @test isapprox(weight(exact.retarded[β]), Id; atol=1e-10)
        end

        # the seed method obeys the identity within the statistical error of the individual samples
        retarded_info = nothing
        method = ThermalSeedBandLanczosMethod(β=2.0, nrandom=50, seed=UInt64(12345), maxdim₁=500, maxdim₂=500)
        gf = GreenFunction(operators, ed, GreenFunctionKind(:retarded), method; e₀=e₀, sector₀=sector₀, callback=info->(retarded_info=info))
        @test retarded_info.Z > 0
        @test retarded_info.Ω ≈ e₀-log(retarded_info.Z)/2.0
        @test Abelian(retarded_info.sector₀) == Abelian(sector₀)
        samples = copy(method.samples)
        @test length(samples) == method.nrandom
        ws = [weight(sample) for sample in samples]
        meanw = sum(ws)/length(ws)
        σ = sqrt(sum(w->sum(abs2, w-meanw), ws)/(length(ws)-1)/length(ws))
        @test maximum(abs.(weight(gf)-Id)) <= max(5σ, 0.05)
    end

    @testset "GreenFunction: parallel construction (ntasks>1)" begin
        # zero-temperature: the parallel sector construction is deterministic and matches the serial one
        serial = (greater=g0, lesser=l0)
        for kind in (GreenFunctionKind(:greater), GreenFunctionKind(:lesser)), keepvecs in (false, true)
            gs = keepvecs ? serial[Symbol(kind)] : GreenFunction(operators, ed, kind, BandLanczosMethod(maxdim=500; keepvecs=false); initial...)
            gp = GreenFunction(operators, ed, kind, BandLanczosMethod(maxdim=500; keepvecs=keepvecs, ntasks=2); initial...)
            @test gs == gp
        end

        # zero-temperature retarded: the greater/lesser parts are constructed concurrently
        parallel_info = nothing
        rp = GreenFunction(operators, ed, GreenFunctionKind(:retarded), BandLanczosMethod(maxdim=500; ntasks=2); initial..., sign=false, callback=info->(parallel_info=info))
        @test r_lanczos0 == rp
        @test parallel_info.Z == 1
        @test parallel_info.Ω ≈ e₀
        @test Abelian(parallel_info.sector₀) == Abelian(sector₀)

        # finite temperature: serial and parallel constructions are bitwise identical for a nonzero seed
        method₁ = ThermalSeedBandLanczosMethod(β=2.0, nrandom=10, seed=UInt64(12345), maxdim₁=500, maxdim₂=500)
        method₂ = ThermalSeedBandLanczosMethod(β=2.0, nrandom=10, seed=UInt64(12345), maxdim₁=500, maxdim₂=500, ntasks=2)
        method₄ = ThermalSeedBandLanczosMethod(β=2.0, nrandom=10, seed=UInt64(12345), maxdim₁=500, maxdim₂=500, ntasks=4)
        gs = GreenFunction(operators, ed, GreenFunctionKind(:greater), method₁; e₀=e₀, sector₀=sector₀)
        gp₁ = GreenFunction(operators, ed, GreenFunctionKind(:greater), method₂; e₀=e₀, sector₀=sector₀)
        gp₂ = GreenFunction(operators, ed, GreenFunctionKind(:greater), method₄; e₀=e₀, sector₀=sector₀)
        @test gs == gp₁ == gp₂
        @test length(method₂.samples) == method₂.nrandom

        # finite-temperature retarded: nested parallelism over samples, parts and sectors
        method₁ = ThermalSeedBandLanczosMethod(β=2.0, nrandom=10, seed=UInt64(777), maxdim₁=500, maxdim₂=500)
        method₂ = ThermalSeedBandLanczosMethod(β=2.0, nrandom=10, seed=UInt64(777), maxdim₁=500, maxdim₂=500, ntasks=2)
        method₄ = ThermalSeedBandLanczosMethod(β=2.0, nrandom=10, seed=UInt64(777), maxdim₁=500, maxdim₂=500, ntasks=4)
        serial_info, nested_info, thermal_info = nothing, nothing, nothing
        rs = GreenFunction(operators, ed, GreenFunctionKind(:retarded), method₁; e₀=e₀, sector₀=sector₀, sign=false, callback=info->(serial_info=info))
        rp₁ = GreenFunction(operators, ed, GreenFunctionKind(:retarded), method₂; e₀=e₀, sector₀=sector₀, sign=false, callback=info->(nested_info=info))
        rp₂ = GreenFunction(operators, ed, GreenFunctionKind(:retarded), method₄; e₀=e₀, sector₀=sector₀, sign=false, callback=info->(thermal_info=info))
        @test rs == rp₁ == rp₂
        @test length(method₂.samples) == method₂.nrandom
        @test serial_info.Z == nested_info.Z == thermal_info.Z
        @test serial_info.Ω == nested_info.Ω == thermal_info.Ω
        for info in (serial_info, nested_info, thermal_info)
            @test info.Z > 0
            @test info.Ω ≈ e₀-log(info.Z)/2.0
            @test Abelian(info.sector₀) == Abelian(sector₀)
        end
    end

    @testset "GreenFunction Hubbard plots" begin
        N, η = 201, 0.1
        es = LinRange(-10.0, 10.0, N)
        path = ReciprocalPath(unitcell, line"X₂-X₁"; length=length(lattice))
        dos, spectral = zeros(N), zeros(N, length(path))
        for (i, e) in enumerate(es)
            data = r_lanczos0(e+η*1im)
            dos[i] = -2imag(tr(data))
            for (m, opₘ) in enumerate(operators), (n, opₙ) in enumerate(operators)
                disp = lattice[opₘ.site] - lattice[opₙ.site]
                for (j, momentum) in enumerate(path)
                    spectral[i, j] += -imag(data[m, n]*exp(1im*dot(disp, momentum)))
                end
            end
        end
        @test isapprox(sum(dos)*(es[2]-es[1]), 2π*length(operators); rtol=0.02)
        Plots.savefig(Plots.plot(es, dos; minorticks=true, minorgrid=true), "Plots-Hubbard-1d-4-DOS.png")
        Makie.save("Makie-Hubbard-1d-4-DOS.png", Makie.lines(es, dos))
        Plots.savefig(Plots.plot(path, es, log.(1 .+ spectral); minorticks=true, minorgrid=true), "Plots-Hubbard-1d-4-spectral.png")
        Makie.save("Makie-Hubbard-1d-4-spectral.png", Makie.plot(path, es, log.(1 .+ spectral)))
    end
end

@testset "GreenFunction: Heisenberg chain" begin
    unitcell = Lattice([0.0]; vectors=[[1.0]])
    lattice = Lattice(unitcell, (8,), ('P',))
    hilbert = Hilbert(Spin{1//2}(), length(lattice))
    J = Heisenberg(:J, 1.0, 1)
    ed = Algorithm(:ED, ED(lattice, hilbert, J, 𝕊ᶻ(0)))
    operators = [𝕊{1//2}(i, '+') for i in eachindex(lattice)]
    ωs = [0.0, 0.5, 1.0, -0.5] .+ 0.05im
    eigensystem = eigen(ed; nev=1)
    e₀, v₀, sector₀ = only(eigensystem.values), only(eigensystem.vectors), only(eigensystem.sectors)
    initial = (e₀=e₀, v₀=v₀, sector₀=sector₀)
    E₀, U₀ = eigen(Hermitian(collect(sparse(ed.frontend, sector₀))))
    thermal = (E₀=E₀, U₀=U₀, sector₀=sector₀)

    g = GreenFunction(operators, ed, GreenFunctionKind(:greater), BandLanczosMethod(; keepvecs=true); initial...)
    expected_weight = sum(operators) do op
        real(dot(v₀, matrix(adjoint(op)*op, (sector₀, sector₀), ed.frontend.matrixization.table, Float64), v₀))
    end
    @test isapprox(tr(g.L*transpose(g.R)), expected_weight; atol=1e-8)

    N, η = 201, 0.1
    es = LinRange(0.0, 4.0, N)
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
    Plots.savefig(Plots.plot(path, es, log.(1 .+ spectral); minorticks=true, minorgrid=true), "Plots-Heisenberg-1d-8-spectral.png")
    Makie.save("Makie-Heisenberg-1d-8-spectral.png", Makie.plot(path, es, log.(1 .+ spectral)))

    # statistical agreement between the seed and exact methods on a spin chain (exercising the AbelianBases path)
    for β in (1.0, 5.0), kind in (GreenFunctionKind(:greater), GreenFunctionKind(:lesser))
        method = ThermalSeedBandLanczosMethod(β=β, nrandom=50, seed=UInt64(12345), maxdim₁=500, maxdim₂=500)
        g_exact = GreenFunction(operators, ed, kind, ThermalExactMethod(β=β); thermal...)
        g_combined = GreenFunction(operators, ed, kind, method; e₀=e₀, sector₀=sector₀)
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
