using ExactDiagonalization
using QuantumLattices: Algorithm, Fock, Hilbert, Hopping, Hubbard, Index, Lattice, Onsite
using QuantumLattices: expand
using SparseArrays: SparseMatrixCSC, nnz, spdiagm
using Distributed: @everywhere, addprocs, remotecall_fetch, rmprocs

@testset "DiskCache" begin
    H₀ = spdiagm(0=>Float64.(1:1000), 1=>ones(999), -1=>ones(999))
    cache = DiskCache(; threshold=0)
    key = "test-key-1"

    # in-memory hit returns the identical object; identity assertions through WeakRef are only meaningful with the GC disabled
    # (Julia's compiler may otherwise reclaim weakly referenced objects earlier than source-level bindings suggest)
    GC.enable(false)
    try
        H = spdiagm(0=>Float64.(1:1000), 1=>ones(999), -1=>ones(999))
        dm = get!(()->H, cache, key)
        @test dm isa DiskCSCMatrix
        @test !isnothing(dm.path)   # threshold=0 forces disk backing
        @test sparse(dm) === H
    finally
        GC.enable(true)
    end

    # GC-driven eviction and transparent disk reload
    function store_and_drop(cache, key)
        dm = get!(()->spdiagm(0=>Float64.(1:1000), 1=>ones(999), -1=>ones(999)), cache, key)
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
    dm = get!(()->H, cache′, key)
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

@testset "DiskCache thread safety" begin
    mkmatrix(i) = spdiagm(0=>Float64.(i:i+19), 1=>ones(19), -1=>ones(19))
    dir = mktempdir()
    cache = DiskCache(; parent=dir, threshold=0)   # threshold=0 forces disk backing
    try
        # concurrent stores of distinct keys: contents correct and backing directories never reused
        Threads.@threads for i = 1:16
            get!(()->mkmatrix(i), cache, "tkey-$i")
        end
        for i = 1:16
            @test sparse(cache.entries["tkey-$i"]) == mkmatrix(i)
        end
        @test allunique([cache.entries["tkey-$i"].path for i = 1:16])

        # concurrent stores of the same key with identical content
        Threads.@threads for i = 1:8
            get!(()->mkmatrix(1), cache, "shared")
        end
        @test sparse(cache.entries["shared"]) == mkmatrix(1)

        # concurrent reload of the same entry after GC eviction
        GC.gc(); GC.gc(); GC.gc()
        @test cache.entries["tkey-1"].payload isa WeakRef && isnothing(cache.entries["tkey-1"].payload.value)
        Threads.@threads for i = 1:16
            sparse(cache.entries["tkey-1"])
        end
        @test sparse(cache.entries["tkey-1"]) == mkmatrix(1)

        # concurrent get!/empty! smoke: no deadlock, surviving entries intact
        Threads.@threads for i = 1:32
            get!(()->mkmatrix(i%16+1), cache, "chaos-$(i%4)")
            i%8==0 && empty!(cache)
        end
        for dm in values(cache.entries)
            @test sparse(dm) isa SparseMatrixCSC
            @test dm.nnz == nnz(sparse(dm)) && (dm.m, dm.n) == size(sparse(dm))
        end

        # lazy global initialization race: all threads get the identical cache
        caches = Vector{Any}(undef, 8)
        Threads.@threads for i = 1:8
            caches[i] = diskcache()
        end
        @test all(c -> c === diskcache(), caches)
    finally
        try
            rm(dir; recursive=true, force=true)
        catch
            # best effort: live memory mappings may hold files on Windows
        end
    end
end

@testset "DiskCache multi-process transfer" begin
    procs = addprocs(2)
    try
        @everywhere using ExactDiagonalization
        @everywhere function remote_materialize(dm)
            return sparse(dm)
        end
        @everywhere function remote_path(dm)
            return dm.path
        end
        M = spdiagm(0=>Float64.(1:100), 1=>ones(99), -1=>ones(99))
        diskcache!(DiskCache(; threshold=0))
        try
            dm = get!(()->M, diskcache(), "xfer")
            @test !isnothing(dm.path)
            for w in procs
                # the proxy is detached on the wire: an in-memory matrix, independent of the sender's files
                @test isnothing(remotecall_fetch(Main.remote_path, w, dm))
                @test remotecall_fetch(Main.remote_materialize, w, dm) == M
            end
            # the transferred copy survives the sender's cache being emptied
            empty!(diskcache())
            @test remotecall_fetch(Main.remote_materialize, procs[1], dm) == M
        finally
            diskcache!(DiskCache())
        end
    finally
        rmprocs(procs)
    end
end

@testset "DiskCache thread-level ED integration" begin
    diskcache!(DiskCache(; threshold=0))
    try
        unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
        lattice = Lattice(unitcell, (4,))
        hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
        terms = (Hopping(:t, -1.0, 1), Hubbard(:U, 4.0), Onsite(:μ, -2.0))
        ed = Algorithm(Symbol("threads"), ED(lattice, hilbert, terms, ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
        sector = only(eigen(ed; nev=1).sectors)

        # concurrent cached matrix construction of the same block: identical shared proxy, no corruption
        ops = expand(ed.frontend.system)
        table = ed.frontend.matrixization.table
        dms = Vector{Any}(undef, 8)
        Threads.@threads for i = 1:8
            dms[i] = matrix(ops, (sector, sector), table; cached=true)
        end
        @test all(dm -> dm === dms[1], dms)

        # type stability of the memoized construction: the hit path returns the shared proxy, the miss path a fresh one
        @test (@inferred matrix(ops, (sector, sector), table; cached=true)) === dms[1]
        @test (@inferred matrix(ops[1], (sector, sector), table; cached=true)) isa DiskCSCMatrix{Float64, Int}

        # concurrent eigen of identical systems shares the blocks and agrees with the serial result
        E = eigen(ed; nev=1).values[1]
        results = Vector{Any}(undef, 4)
        Threads.@threads for i = 1:4
            edi = Algorithm(Symbol("threads-$i"), ED(lattice, hilbert, terms, ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)))
            results[i] = eigen(edi; nev=1).values[1]
        end
        @test all(≈(E), results)
    finally
        diskcache!(DiskCache())
    end
end
