using ExactDiagonalization
using QuantumLattices: Algorithm, BrillouinZone, Coulomb, Fock, Heisenberg, Hilbert, Hopping, Hubbard, Lattice, Metric, ReciprocalPath, ReciprocalZone, Spin, Table, Zeeman
using QuantumLattices: σ⁺, σ⁻, σᶻ, 𝕔, 𝕔⁺, 𝕔⁺𝕔, bonds, expand, @hexagon_str, @rectangle_str
import CairoMakie as Makie
import Plots

@testset "Square lattice static structure factors" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))

    @testset "SpinlessSquareStaticChargeStructFactor" begin
        hilbert = Hilbert(Fock{:f}(1, 1), length(lattice))
        ed = Algorithm(Symbol("Spinless-Square-2x2"), ED(lattice, hilbert, (Hopping(:t, -1.0, 1), Coulomb(:V, 2.0, 1)), ℕ(length(lattice)÷2)))
        eigensystem = ed(:eigen, EDEigen(); delay=true)
        nᵢ = [𝕔⁺(i, 1, 0)*𝕔(i, 1, 0) for i = 1:length(lattice)]
        expectation = ed(Symbol("Spinless-Square-2x2-GroundStateExpectation"), GroundStateExpectation(nᵢ), eigensystem)
        @test sum(expectation.data.values) ≈ length(lattice)÷2
        nᵢnⱼ = [(nᵢ[i]-expectation.data.values[i])*((nᵢ[j]-expectation.data.values[j])) for i=1:length(lattice), j=1:length(lattice)]
        bz = ed(Symbol("Spinless-Square-2x2-StaticChargeStructureFactor-BZ"), StaticTwoPointCorrelator(nᵢnⱼ, BrillouinZone(unitcell, 21)), eigensystem)
        Plots.savefig(Plots.plot(bz), "Plots-Spinless-Square-2x2-StaticChargeStructureFactor-BZ.png")
        Makie.save("Makie-Spinless-Square-2x2-StaticChargeStructureFactor-BZ.png", Makie.plot(bz))
        path = ed(Symbol("Spinless-Square-2x2-StaticChargeStructureFactor-Path"), StaticTwoPointCorrelator(nᵢnⱼ, ReciprocalPath(unitcell, rectangle"Γ-X-M-Γ"; length=41)), eigensystem)
        @test isapprox(path.data.values[1], 0.0; atol=1e-10)
        Plots.savefig(Plots.plot(path), "Plots-Spinless-Square-2x2-StaticChargeStructureFactor-Path.png")
        Makie.save("Makie-Spinless-Square-2x2-StaticChargeStructureFactor-Path.png", Makie.plot(path))
    end

    @testset "HubbardSquareStaticSpinStructureFactor" begin
        hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
        ed = Algorithm(Symbol("Hubbard-Square-2x2"), ED(lattice, hilbert, (Hopping(:t, -1.0, 1), Hubbard(:U, 2.0)), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
        eigensystem = ed(:eigen, EDEigen(); delay=true)
        SᵢSⱼ = [expand(Coulomb(:V, 1//4, :, 1//2*𝕔⁺𝕔(:, :, σ⁺)*𝕔⁺𝕔(:, :, σ⁻) + 1//2*𝕔⁺𝕔(:, :, σ⁻)*𝕔⁺𝕔(:, :, σ⁺) + 𝕔⁺𝕔(:, :, σᶻ)*𝕔⁺𝕔(:, :, σᶻ)), bond, hilbert) for bond in bonds(lattice, :)]
        bz = ed(Symbol("Hubbard-Square-2x2-StaticSpinStructureFactor-BZ"), StaticTwoPointCorrelator(SᵢSⱼ, BrillouinZone(unitcell, 21)), eigensystem)
        Plots.savefig(Plots.plot(bz), "Plots-Hubbard-Square-2x2-StaticSpinStructureFactor-BZ.png")
        Makie.save("Makie-Hubbard-Square-2x2-StaticSpinStructureFactor-BZ.png", Makie.plot(bz))
        path = ed(Symbol("Hubbard-Square-2x2-StaticSpinStructureFactor-Path"), StaticTwoPointCorrelator(SᵢSⱼ, ReciprocalPath(unitcell, rectangle"Γ-X-M-Γ"; length=41)), eigensystem)
        @test isapprox(path.data.values[1], 0.0; atol=1e-8)
        Plots.savefig(Plots.plot(path), "Plots-Hubbard-Square-2x2-StaticSpinStructureFactor-Path.png")
        Makie.save("Makie-Hubbard-Square-2x2-StaticSpinStructureFactor-Path.png", Makie.plot(path))
    end

    @testset "HeisenbergSquareStaticSpinStructureFactor" begin
        hilbert = Hilbert(Spin{1//2}(), length(lattice))
        ed = Algorithm(Symbol("Heisenberg-Square-2x2"), ED(lattice, hilbert, Heisenberg(:J, 1.0, 1), 𝕊ᶻ(0)))
        eigensystem = ed(:eigen, EDEigen(); delay=true)
        SᵢSⱼ = [expand(Heisenberg(:J, 1.0, :), bond, hilbert) for bond in bonds(lattice, :)]
        bz = ed(Symbol("Heisenberg-Square-2x2-SpinStructureFactor-BZ"), StaticTwoPointCorrelator(SᵢSⱼ, BrillouinZone(unitcell, 21)), eigensystem)
        Plots.savefig(Plots.plot(bz), "Plots-Heisenberg-Square-2x2-SpinStructureFactor-BZ.png")
        Makie.save("Makie-Heisenberg-Square-2x2-SpinStructureFactor-BZ.png", Makie.plot(bz))
        path = ed(Symbol("Heisenberg-Square-2x2-SpinStructureFactor-Path"), StaticTwoPointCorrelator(SᵢSⱼ, ReciprocalPath(unitcell, rectangle"Γ-X-M-Γ"; length=41)), eigensystem)
        @test isapprox(path.data.values[1], 0.0; atol=1e-8)
        Plots.savefig(Plots.plot(path), "Plots-Heisenberg-Square-2x2-StaticSpinStructureFactor-Path.png")
        Makie.save("Makie-Heisenberg-Square-2x2-StaticSpinStructureFactor-Path.png", Makie.plot(path))
    end
end

@testset "HeisenbergHexagonStaticSpinStructureFactor" begin
    unitcell = Lattice([0.0, 0.0], [0.0, √3/3]; vectors=[[1.0, 0.0], [0.5, √3/2]])
    lattice = Lattice(
        [0.0, 0.0], [0.0, √3/3], [0.5, √3/2], [0.5, -√3/6], [1.0, 0.0], [1.0, √3/3];
        vectors=[[1.5, √3/2], [1.5, -√3/2]]
    )
    hilbert = Hilbert(Spin{1//2}(), length(lattice))
    ed = Algorithm(Symbol("Hexagon-H6"), ED(lattice, hilbert, Heisenberg(:J, 1.0, 1), 𝕊ᶻ(0)))
    eigensystem = ed(:eigen, EDEigen(); delay=true)
    SᵢSⱼ = [expand(Heisenberg(:J, 1.0, :), bond, hilbert) for bond in bonds(lattice, :)]
    rz = ed(Symbol("Heisenberg-Hexagon-H6-SpinStructureFactor-RZ"), StaticTwoPointCorrelator(SᵢSⱼ, ReciprocalZone([[pi, 0.0], [0.0, pi]], -4=>4, -4=>4)), eigensystem)
    Plots.savefig(Plots.plot(rz), "Plots-Heisenberg-Hexagon-H6-SpinStructureFactor-RZ.png")
    Makie.save("Makie-Heisenberg-Hexagon-H6-SpinStructureFactor-RZ.png", Makie.plot(rz))
    path = ed(Symbol("Heisenberg-Hexagon-H6-SpinStructureFactor-Path"), StaticTwoPointCorrelator(SᵢSⱼ, ReciprocalPath(unitcell, (0, 0), (1, 0), (2, 1); labels=("Γ", "Γ′", "Γ′′"))), eigensystem)
    Plots.savefig(Plots.plot(path), "Plots-Heisenberg-Hexagon-H6-SpinStructureFactor-Path.png")
    Makie.save("Makie-Heisenberg-Hexagon-H6-SpinStructureFactor-Path.png", Makie.plot(path))
end

@testset "SpinCoherentState" begin
    @test SpinCoherentState(Dict(1=>[0, 0, 1], 2=>[0, 0, -1])) == SpinCoherentState(Dict(1=>(0, 0), 2=>(pi, 0)); unit=:radian) == SpinCoherentState(Dict(1=>(0, 0), 2=>(180, 0)); unit=:degree)

    hilbert = Hilbert(Spin{1//2}(), 4)
    table = Table(hilbert, Metric(EDKind(hilbert), hilbert))
    sector = Sector(𝕊ᶻ(0), hilbert)
    @test SpinCoherentState(Dict(i=>i∈(1, 2) ? [0, 0, -1] : [0, 0, 1] for i=1:4))(sector, table) == [1, 0, 0, 0, 0, 0]
    @test SpinCoherentState(Dict(i=>i∈(2, 4) ? [0, 0, -1] : [0, 0, 1] for i=1:4))(sector, table) == [0, 1, 0, 0, 0, 0]
    @test SpinCoherentState(Dict(i=>i∈(2, 3) ? [0, 0, -1] : [0, 0, 1] for i=1:4))(sector, table) == [0, 0, 1, 0, 0, 0]
    @test SpinCoherentState(Dict(i=>i∈(1, 4) ? [0, 0, -1] : [0, 0, 1] for i=1:4))(sector, table) == [0, 0, 0, 1, 0, 0]
    @test SpinCoherentState(Dict(i=>i∈(1, 3) ? [0, 0, -1] : [0, 0, 1] for i=1:4))(sector, table) == [0, 0, 0, 0, 1, 0]
    @test SpinCoherentState(Dict(i=>i∈(3, 4) ? [0, 0, -1] : [0, 0, 1] for i=1:4))(sector, table) == [0, 0, 0, 0, 0, 1]
end

@testset "SpinCoherentStateProjection" begin
    unitcell = Lattice([0.0, 0.0], [0.0, √3/3]; vectors=[[1.0, 0.0], [0.5, √3/2]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    hilbert = Hilbert(Spin{1//2}(), length(lattice))

    coherent = SpinCoherentState(Dict(i=>[0.0, 0.0, 1.0] for i=1:length(lattice)))
    ed = Algorithm(Symbol("Hexagon-2x2"), ED(lattice, hilbert, (Heisenberg(:J, -1.0, 1), Zeeman(:h, Complex(-0.2), (pi/2, pi); unit=:radian))))
    eigensystem = ed(:eigen, EDEigen(); delay=true)
    projection = ed(Symbol("Hexagon-2x2-SpinCoherentState-FM-(90, 180)"), SpinCoherentStateProjection(coherent, 41, 41), eigensystem)
    @test all(x->0.0<=x<=1.0+1e-12, projection.data.values)
    peak = argmax(projection.data.values)
    @test projection.data.polars[peak[2]] ≈ pi/2
    @test projection.data.azimuths[peak[1]] ≈ pi
    Plots.savefig(Plots.plot(projection), "Plots-Hexagon-2x2-SpinCoherentState-FM-(90, 180).png")
    Makie.save("Makie-Hexagon-2x2-SpinCoherentState-FM-(90, 180).png", Makie.plot(projection))

    coherent = SpinCoherentState(Dict(i=>iseven(i) ? [0.0, 0.0, 1.0] : [0.0, 0.0, -1.0] for i=1:length(lattice)))
    ed = Algorithm(Symbol("Hexagon-2x2"), ED(lattice, hilbert, (Heisenberg(:J, 1.0, 1), Zeeman(:h, Complex(-0.2), 'z'; amplitude=bond->iseven(bond[1].site) ? 1 : -1)), 𝕊ᶻ(0)))
    eigensystem = ed(:eigen, EDEigen(); delay=true)
    projection = ed(Symbol("Hexagon-2x2-SpinCoherentState-AFM-z"), SpinCoherentStateProjection(coherent, 41, 41), eigensystem)
    @test all(x->0.0<=x<=1.0+1e-12, projection.data.values)
    reference = coherent(only(eigensystem.data.sectors), ed.frontend.matrixization.table)
    @test projection.data.values[1, 1] ≈ abs2(reference' * only(eigensystem.data.vectors))
    Plots.savefig(Plots.plot(projection), "Plots-Hexagon-2x2-SpinCoherentState-AFM-z.png")
    Makie.save("Makie-Hexagon-2x2-SpinCoherentState-AFM-z.png", Makie.plot(projection))

    ed = Algorithm(Symbol("Hexagon-2x2"), ED(lattice, hilbert, (Heisenberg(:J, 1.0, 1), Zeeman(:h, Complex(-0.2), 'x'; amplitude=bond->iseven(bond[1].site) ? 1 : -1))))
    eigensystem = ed(:eigen, EDEigen(); delay=true)
    projection = ed(Symbol("Hexagon-2x2-SpinCoherentState-AFM-x"), SpinCoherentStateProjection(coherent, 41, 41), eigensystem)
    @test all(x->0.0<=x<=1.0+1e-12, projection.data.values)
    reference = coherent(only(eigensystem.data.sectors), ed.frontend.matrixization.table)
    @test projection.data.values[1, 1] ≈ abs2(reference' * only(eigensystem.data.vectors))
    Plots.savefig(Plots.plot(projection), "Plots-Hexagon-2x2-SpinCoherentState-AFM-x.png")
    Makie.save("Makie-Hexagon-2x2-SpinCoherentState-AFM-x.png", Makie.plot(projection))
end
