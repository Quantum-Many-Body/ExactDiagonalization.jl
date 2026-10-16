# DiskCache: session-scoped disk-backed cache for sparse matrices, with GC-driven eviction

"""
    DiskCSCMatrix{V, I<:Integer} <: AbstractSparseMatrix{V, I}

A sparse matrix in CSC format whose payload is either held strongly in memory (`isnothing(path)`, for small or uncached matrices, never evicted) or held weakly with backing files on disk: in the latter case the garbage collector may reclaim the payload under memory pressure, and the next access transparently reloads it via memory mapping.

Only read-only access is supported; no mutating methods are defined, so accidental in-place modification of a cached matrix is a `MethodError` rather than silent corruption of the backing files.

`sparse` reload is thread-safe (guarded by the per-entry `lock`). When a `DiskCSCMatrix` is serialized to another process, the payload is materialized and the received object is a detached in-memory proxy (`path=nothing`): the session-local backing files never leak across process boundaries.
"""
mutable struct DiskCSCMatrix{V, I<:Integer} <: AbstractSparseMatrix{V, I}
    const lock::ReentrantLock
    const path::Union{Nothing, String}
    const m::Int
    const n::Int
    const nnz::Int
    payload::Union{SparseMatrixCSC{V, I}, WeakRef}
    DiskCSCMatrix{V, I}(path, m, n, nnz, payload) where {V, I<:Integer} = new{V, I}(ReentrantLock(), path, m, n, nnz, payload)
end
# minimal read-only interface
@inline Base.size(dm::DiskCSCMatrix) = (dm.m, dm.n)
@inline nnz(dm::DiskCSCMatrix) = dm.nnz
@inline Base.iszero(dm::DiskCSCMatrix) = dm.nnz==0 || iszero(sparse(dm))
@inline Base.getindex(dm::DiskCSCMatrix, i::Integer, j::Integer) = sparse(dm)[i, j]
@inline Base.:(==)(a::DiskCSCMatrix, b::DiskCSCMatrix) = (a.m, a.n, a.nnz)==(b.m, b.n, b.nnz) && sparse(a)==sparse(b)
@inline Base.:(==)(a::DiskCSCMatrix, b::SparseMatrixCSC) = sparse(a)==b
@inline Base.:(==)(a::SparseMatrixCSC, b::DiskCSCMatrix) = a==sparse(b)
@inline Base.isequal(a::DiskCSCMatrix, b::DiskCSCMatrix) = a==b
# arithmetic: results are fresh anonymous in-memory proxies so that payload types stay uniform; algebraic results are never cached nor written to disk (memoization happens only at the `cached=true` entry points of `matrix`)
@inline Base.:*(dm::DiskCSCMatrix, v::AbstractVector) = sparse(dm)*v
@inline Base.:*(dm::DiskCSCMatrix, m::AbstractMatrix) = sparse(dm)*m
@inline Base.:*(a::Number, dm::DiskCSCMatrix) = DiskCSCMatrix(a*sparse(dm))
@inline Base.:*(dm::DiskCSCMatrix, a::Number) = DiskCSCMatrix(sparse(dm)*a)
@inline Base.:+(a::DiskCSCMatrix, b::DiskCSCMatrix) = DiskCSCMatrix(sparse(a)+sparse(b))
@inline Base.:+(a::DiskCSCMatrix, b::SparseMatrixCSC) = DiskCSCMatrix(sparse(a)+b)
@inline Base.:+(a::SparseMatrixCSC, b::DiskCSCMatrix) = DiskCSCMatrix(a+sparse(b))
@inline Base.:-(a::DiskCSCMatrix, b::DiskCSCMatrix) = a+DiskCSCMatrix(-sparse(b))
@inline Base.:-(dm::DiskCSCMatrix) = DiskCSCMatrix(-sparse(dm))
@inline Base.adjoint(dm::DiskCSCMatrix) = DiskCSCMatrix(adjoint(sparse(dm)))
# fast paths that must reach the materialized payload instead of the generic O(dim²) getindex-based fallbacks
@inline dot(x::AbstractVector, dm::DiskCSCMatrix, y::AbstractVector) = dot(x, sparse(dm), y)
@inline mul!(y::AbstractVecOrMat, dm::DiskCSCMatrix, x::AbstractVecOrMat) = mul!(y, sparse(dm), x)
@inline Base.show(io::IO, dm::DiskCSCMatrix) = print(io, "DiskCSCMatrix(", dm.m, "×", dm.n, ", nnz=", dm.nnz, ", ", isnothing(dm.path) ? "memory" : "disk", ")")
# the detailed plain-text display is delegated to the payload, so a proxy prints exactly like the materialized sparse matrix
@inline Base.show(io::IO, mime::MIME"text/plain", dm::DiskCSCMatrix) = show(io, mime, sparse(dm))

"""
    sparse(dm::DiskCSCMatrix) -> SparseMatrixCSC

Get the payload of a `DiskCSCMatrix` as a plain `SparseMatrixCSC`, extending [`SparseArrays.sparse`](https://sparsearrays.juliasparse.org/dev/#SparseArrays.sparse). If the payload has been garbage-collected, it is transparently reloaded from disk via memory mapping (without copying).

The returned matrix is the shared payload rather than a copy, so it must not be mutated in place.

Thread-safe: the reload (mmap + `WeakRef` swap) is serialized by the per-entry lock; the fast paths (strongly held or still-alive payload) are lock-free.
"""
function sparse(dm::DiskCSCMatrix{V, I})::SparseMatrixCSC{V, I} where {V, I<:Integer}
    payload = dm.payload
    payload isa SparseMatrixCSC && return payload
    value = payload.value
    isnothing(value) || return value
    lock(dm.lock)
    try
        payload = dm.payload
        payload isa SparseMatrixCSC && return payload
        value = payload.value
        isnothing(value) || return value
        value = SparseMatrixCSC{V, I}(
            dm.m, dm.n,
            mmap(joinpath(dm.path, "colptr.bin"), Vector{I}, (dm.n+1,)),
            mmap(joinpath(dm.path, "rowval.bin"), Vector{I}, (dm.nnz,)),
            mmap(joinpath(dm.path, "nzval.bin"), Vector{V}, (dm.nnz,))
        )
        dm.payload = WeakRef(value)
        return value
    finally
        unlock(dm.lock)
    end
end

"""
    serialize(s::AbstractSerializer, dm::DiskCSCMatrix)

Serialize a `DiskCSCMatrix` for cross-process transfer, extending `Serialization.serialize`: the payload is materialized and only the dimensions and the content are written, so the received object is a detached in-memory proxy (`path=nothing`). `Serialization.serialize_cycle` keeps the object identity of shared proxies within one serialized graph.
"""
function serialize(s::AbstractSerializer, dm::DiskCSCMatrix)
    Serialization.serialize_cycle(s, dm) && return nothing
    Serialization.serialize_type(s, typeof(dm), true)
    serialize(s, dm.m)
    serialize(s, dm.n)
    serialize(s, dm.nnz)
    serialize(s, sparse(dm))
    return nothing
end

"""
    deserialize(s::AbstractSerializer, ::Type{DiskCSCMatrix{V, I}}) where {V, I} -> DiskCSCMatrix{V, I}

Deserialize a `DiskCSCMatrix` received from another process, extending `Serialization.deserialize`: the result is a detached in-memory proxy (`path=nothing`), independent of the sender's backing files (see [`serialize`](@ref)).
"""
function deserialize(s::AbstractSerializer, ::Type{DiskCSCMatrix{V, I}}) where {V, I}
    m = deserialize(s)::Int
    n = deserialize(s)::Int
    nz = deserialize(s)::Int
    M = deserialize(s)::SparseMatrixCSC{V, I}
    dm = DiskCSCMatrix{V, I}(nothing, m, n, nz, M)
    Serialization.deserialize_cycle(s, dm)
    return dm
end

"""
    DiskCSCMatrix(M::SparseMatrixCSC) -> DiskCSCMatrix

Wrap a sparse matrix into an in-memory `DiskCSCMatrix` proxy (strongly held, never evicted, nothing written to disk, not entered into any cache).
"""
@inline DiskCSCMatrix(M::SparseMatrixCSC{V, I}) where {V, I} = DiskCSCMatrix{V, I}(nothing, M.m, M.n, nnz(M), M)

"""
    DiskCache

A session-scoped disk-backed store of sparse matrices ([`DiskCSCMatrix`](@ref)) keyed by serialized content (see [`cachekey`](@ref)), with GC-driven eviction and transparent reload. Since the `Dict` keys are the full content bytes, distinct contents can never be confused, regardless of hash collisions. The on-disk directories are named by a monotonic counter: deduplication is the dictionary's job, not the file names'.

Entries are accessed exclusively through [`get!`](@ref) with a lazy default (`get!(f, cache, key)`: `f` produces the matrix only on a miss) and removed through [`empty!`](@ref). Thread-safe: they are serialized by the per-cache `lock`, and the disk writes of a new entry happen outside the lock under a directory name that is unique by construction. Process-safe by isolation: each process (and each `DiskCache` instance) writes to its own fresh temporary directory, so no cross-process coordination is needed.
"""
mutable struct DiskCache
    const lock::ReentrantLock
    const dir::String
    const threshold::Int
    const entries::Dict{String, DiskCSCMatrix}
    counter::Int
    DiskCache(dir, threshold, entries, counter) = new(ReentrantLock(), dir, threshold, entries, counter)
end

"""
    DiskCache(; parent::AbstractString=tempdir(), threshold::Integer=1<<20) -> DiskCache

Create a `DiskCache` whose files live in a fresh subdirectory of `parent` (use a fast local disk, e.g. an NVMe scratch area, for `parent` when available; on Unix the default `tempdir()` honors the `TMPDIR` environment variable). Matrices smaller than `threshold` bytes are wrapped in in-memory proxies and never written to disk. The directory is removed (best effort) at the end of the session. The on-disk format is native binary and valid only within the current session.
"""
function DiskCache(; parent::AbstractString=tempdir(), threshold::Integer=1<<20)
    dir = mktempdir(parent)
    atexit() do
        try rm(dir; recursive=true, force=true) catch end
    end
    return DiskCache(dir, threshold, Dict{String, DiskCSCMatrix}(), 0)
end

"""
    get!(f::Function, cache::DiskCache, key::String) -> DiskCSCMatrix

Get the proxy cached under `key`; on a miss, invoke `f()` to produce the matrix, store it under `key`, and return its proxy, extending `Base.get!`. This is the sole entry accessor of `DiskCache`: the returned value is always the proxy associated with `key` after the call, so two concurrent calls with the same key both return the winner's proxy. Matrices smaller than `cache.threshold` bytes are wrapped in an in-memory proxy (strongly held, never evicted) and are not written to disk.

Thread-safe: the key lookup, the allocation of the on-disk directory name, and the registration of the new entry happen under the per-cache lock, while `f()` and the disk writes happen outside it. Two concurrent calls with the same key may both invoke `f()` (the expensive construction is deliberately not serialized); the loser discards its files and returns the winner's proxy.
"""
@inline function Base.get!(f::Function, cache::DiskCache, key::String)
    existing = lock(cache.lock) do
        get(cache.entries, key, nothing)
    end
    isnothing(existing) || return existing
    return store!(cache, key, f())
end
# miss path of `get!`: wrap the freshly constructed matrix `M` and register it under `key`; the function barrier keeps the concrete payload type `DiskCSCMatrix{V, I}` on the dispatch level
function store!(cache::DiskCache, key::String, M::SparseMatrixCSC{V, I})::DiskCSCMatrix{V, I} where {V, I}
    ondisk = sizeof(M.colptr)+sizeof(M.rowval)+sizeof(M.nzval) >= cache.threshold
    dir = lock(cache.lock) do
        ondisk ? joinpath(cache.dir, string(cache.counter += 1, base=16)) : nothing
    end
    local dm::DiskCSCMatrix{V, I}
    if isnothing(dir)
        dm = DiskCSCMatrix(M)
    else
        mkpath(dir)
        open(joinpath(dir, "header"), "w") do io
            print(io, string(V), " ", string(I), " ", M.m, " ", M.n, " ", nnz(M))
        end
        write(joinpath(dir, "colptr.bin"), M.colptr)
        write(joinpath(dir, "rowval.bin"), M.rowval)
        write(joinpath(dir, "nzval.bin"), M.nzval)
        dm = DiskCSCMatrix{V, I}(dir, M.m, M.n, nnz(M), WeakRef(M))
    end
    return lock(cache.lock) do
        existing = get(cache.entries, key, nothing)
        if isnothing(existing)
            cache.entries[key] = dm
            dm
        else
            isnothing(dir) || try rm(dir; recursive=true, force=true) catch end
            existing::DiskCSCMatrix{V, I}
        end
    end
end

"""
    empty!(cache::DiskCache) -> DiskCache

Remove all entries of `cache` and delete their files (best effort), extending `Base.empty!`. The directory-name counter is deliberately not reset, so that directories whose deletion failed (e.g. due to live memory mappings) are never reused for new contents.

Thread-safe: the entry table is cleared under the per-cache lock, and the files of each entry are deleted under the per-entry lock so that a concurrent `sparse` reload cannot race the deletion.
"""
function Base.empty!(cache::DiskCache)
    entries = lock(cache.lock) do
        result = collect(values(cache.entries))
        empty!(cache.entries)
        result
    end
    for entry in entries
        isnothing(entry.path) && continue
        lock(entry.lock)
        try
            rm(entry.path; recursive=true, force=true)
        catch
        finally
            unlock(entry.lock)
        end
    end
    return cache
end

const DISKCACHE = Ref{Union{Nothing, DiskCache}}(nothing)
const DISKCACHE_LOCK = ReentrantLock()
"""
    diskcache() -> DiskCache

Get the process-wide default [`DiskCache`](@ref), creating it lazily on first use. Since the cache is keyed by serialized content, sharing it across different systems is safe: identical contents map to identical matrices, and distinct contents are never confused.

Thread-safe: the lazy initialization is serialized by a global lock.
"""
function diskcache()
    return lock(DISKCACHE_LOCK) do
        cache = DISKCACHE[]
        isnothing(cache) ? (DISKCACHE[] = DiskCache()) : cache
    end
end

"""
    diskcache!(cache::DiskCache) -> DiskCache

Set the process-wide default disk cache (e.g., to customize the parent directory of its files or the size threshold). The previous default is discarded (its directory is removed on a best-effort basis).

Thread-safe: the swap of the default is atomic; the previous default is emptied afterwards.
"""
function diskcache!(cache::DiskCache)
    old = lock(DISKCACHE_LOCK) do
        old = DISKCACHE[]
        DISKCACHE[] = cache
        old
    end
    isnothing(old) || empty!(old)
    return cache
end
