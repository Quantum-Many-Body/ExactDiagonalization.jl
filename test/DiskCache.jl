using ExactDiagonalization
using QuantumLattices
using SparseArrays

@testset "DiskCache" begin
    H₀ = spdiagm(0=>Float64.(1:1000), 1=>ones(999), -1=>ones(999))
    cache = DiskCache(; threshold=0)
    key = "test-key-1"

    # in-memory hit returns the identical object; identity assertions through WeakRef are only meaningful with the GC disabled
    # (Julia's compiler may otherwise reclaim weakly referenced objects earlier than source-level bindings suggest)
    GC.enable(false)
    try
        H = spdiagm(0=>Float64.(1:1000), 1=>ones(999), -1=>ones(999))
        dm = ExactDiagonalization.store!(cache, key, H)
        @test dm isa DiskCSCMatrix
        @test !isnothing(dm.path)   # threshold=0 forces disk backing
        @test sparse(dm) === H
    finally
        GC.enable(true)
    end

    # GC-driven eviction and transparent disk reload
    function store_and_drop(cache, key)
        dm = ExactDiagonalization.store!(cache, key, spdiagm(0=>Float64.(1:1000), 1=>ones(999), -1=>ones(999)))
        return dm.path
    end
    path = store_and_drop(cache, "test-key-2")
    GC.gc()
    GC.gc()
    GC.gc()
    dm = cache.entries["test-key-2"]
    dead = dm.payload isa WeakRef && isnothing(dm.payload.value)
    @test dead
    H = sparse(dm)
    @test H !== H₀
    @test H == H₀
    @test nnz(H) == nnz(H₀)
    @test size(dm) == size(H₀)
    @test nnz(dm) == nnz(H₀)

    # matrices below the threshold are wrapped in in-memory proxies and never written to disk
    cache′ = DiskCache()
    dm = ExactDiagonalization.store!(cache′, key, H)
    @test isnothing(dm.path)
    @test sparse(dm) == H₀
end

@testset "DiskCache integration with ED" begin
    # integration with ED via the process-wide default cache: hits, and key separation across update!
    diskcache!(DiskCache(; threshold=0))
    try
        unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
        lattice = Lattice(unitcell, (4,))
        hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
        t = Hopping(:t, -1.0, 1)
        U = Hubbard(:U, 4.0)
        μ = Onsite(:μ, -2.0)
        ed = Algorithm(Symbol("1D"), ED(lattice, hilbert, (t, U, μ), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
        sector₀ = only(eigen(ed; nev=1).sectors)
        GC.enable(false)
        try
            H = sparse(ed.frontend, sector₀)
            cached = sparse(ed.frontend, sector₀)
            @test cached == H
        finally
            GC.enable(true)
        end
        @test !isempty(diskcache().entries)

        # update! changes the cache key: the reloaded matrix must follow the new parameters
        H₁ = sparse(ed.frontend, sector₀)
        update!(ed; t=-2.0)
        H₂ = sparse(ed.frontend, sector₀)
        @test H₂ ≠ H₁
        ed′ = Algorithm(Symbol("1D"), ED(lattice, hilbert, (Hopping(:t, -2.0, 1), U, μ), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
        @test H₂ == sparse(ed′.frontend, only(eigen(ed′; nev=1).sectors))

        # identical Green functions with in-memory and reloaded-from-disk matrices
        operators = mapreduce(vcat, hilbert) do pair
            site, fock = pair
            return [Index(site, fock[i]) for i=1:length(fock)÷2]
        end
        eigensystem = eigen(ed′; nev=1)
        e₀, v₀, s₀ = only(eigensystem.values), only(eigensystem.vectors), only(eigensystem.sectors)
        g₁ = GreenFunction(operators, ed′, GreenFunctionKind(:greater), BandLanczosMethod(); e₀=e₀, v₀=v₀, sector₀=s₀)
        GC.gc()
        GC.gc()
        g₂ = GreenFunction(operators, ed′, GreenFunctionKind(:greater), BandLanczosMethod(); e₀=e₀, v₀=v₀, sector₀=s₀)
        @test g₁ == g₂

        # Operators-level cache key: `cached=true` registers and shares the proxy; the default construction is anonymous
        ops = expand(ed′.frontend.system)
        braket = (s₀, s₀)
        table = ed′.frontend.matrixization.table
        a = matrix(ops, braket, table; cached=true)
        b = matrix(ops, braket, table; cached=true)
        @test a === b
        c = matrix(ops, braket, table)
        @test c !== a
        @test c == a

        # block-level memoization: identical systems share the identical proxy payloads in ed.H
        blocks(ed) = [m.matrix for ops in values(ed.frontend.H.alterops) for m in ops]
        ed₁ = Algorithm(Symbol("1D"), ED(lattice, hilbert, (Hopping(:t, -3.0, 1), U, μ), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
        ed₂ = Algorithm(Symbol("1D"), ED(lattice, hilbert, (Hopping(:t, -3.0, 1), U, μ), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
        E = eigen(ed₁; nev=1).values[1]
        eigen(ed₂; nev=1)
        bs₁ = blocks(ed₁)
        bs₂ = blocks(ed₂)
        @test length(bs₁) == length(bs₂) > 0
        @test all(bs₁ .=== bs₂)

        # release! frees ed.H and the payloads are GC-evicted; the next construction transparently reloads from disk
        release!(ed₁)
        GC.gc()
        GC.gc()
        GC.gc()
        @test all(dm->dm.payload isa WeakRef && isnothing(dm.payload.value), bs₁)
        @test eigen(ed₁; nev=1).values[1] ≈ E

        # update! does not touch the parameter-independent cached blocks
        update!(ed₁; t=-4.0)
        @test all(blocks(ed₁) .=== bs₁)
        ed₃ = Algorithm(Symbol("1D"), ED(lattice, hilbert, (Hopping(:t, -4.0, 1), U, μ), ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
        @test eigen(ed₁; nev=1).values[1] ≈ eigen(ed₃; nev=1).values[1]
    finally
        diskcache!(DiskCache())
    end
end
