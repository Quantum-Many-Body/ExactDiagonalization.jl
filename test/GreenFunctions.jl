using ExactDiagonalization
using LinearAlgebra: tr
using Plots: plot, savefig
using QuantumLattices

@testset "GreenFunction" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 5))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 8.0)
    μ = Onsite(:μ, -U.value/2)

    ed = Algorithm(Symbol("1D"), ED(lattice, hilbert, (t, U, μ), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
    operators = mapreduce(vcat, hilbert) do pair
        site, fock = pair
        return [Index(site, fock[i]) for i=1:length(fock)÷2]
    end
    g = RetardedGreenFunction(operators, ed; maxdim=100)

    emin = -10.0
    emax = 10.0
    N = 501
    η = 0.05
    es = LinRange(emin, emax, N)
    dos = zeros(N)
    for (i, e) in enumerate(es)
        dos[i] = -2imag(tr(g(e+η*1im)))
    end
    savefig(plot(es, dos; minorticks=true, minorgrid=true), "Hubbard-Square-2x5-DOS.png")
end

