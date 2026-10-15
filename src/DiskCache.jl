# DiskCache: session-scoped disk-backed cache for sparse matrices, with GC-driven eviction

"""
    DiskCSCMatrix{V, I<:Integer} <: AbstractSparseMatrix{V, I}

A sparse matrix in CSC format whose payload is either held strongly in memory (`isnothing(path)`, for small or uncached matrices, never evicted) or held weakly with backing files on disk: in the latter case the garbage collector may reclaim the payload under memory pressure, and the next access transparently reloads it via memory mapping.

Only read-only access is supported; no mutating methods are defined, so accidental in-place modification of a cached matrix is a `MethodError` rather than silent corruption of the backing files.
"""
mutable struct DiskCSCMatrix{V, I<:Integer} <: AbstractSparseMatrix{V, I}
    const path::Union{Nothing, String}
    const m::Int
    const n::Int
    const nnz::Int
    payload::Union{SparseMatrixCSC{V, I}, WeakRef}
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
"""
function sparse(dm::DiskCSCMatrix{V, I})::SparseMatrixCSC{V, I} where {V, I<:Integer}
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
end

"""
    DiskCSCMatrix(M::SparseMatrixCSC) -> DiskCSCMatrix

Wrap a sparse matrix into an in-memory `DiskCSCMatrix` proxy (strongly held, never evicted, nothing written to disk, not entered into any cache).
"""
@inline DiskCSCMatrix(M::SparseMatrixCSC{V, I}) where {V, I} = DiskCSCMatrix{V, I}(nothing, M.m, M.n, nnz(M), M)

"""
    DiskCache

A session-scoped disk-backed store of sparse matrices ([`DiskCSCMatrix`](@ref)) keyed by serialized content (see [`cachekey`](@ref)), with GC-driven eviction and transparent reload. Since the `Dict` keys are the full content bytes, distinct contents can never be confused, regardless of hash collisions. The on-disk directories are named by a monotonic counter: deduplication is the dictionary's job, not the file names'.
"""
mutable struct DiskCache
    const dir::String
    const threshold::Int
    const entries::Dict{String, DiskCSCMatrix}
    counter::Int
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
    store!(cache::DiskCache, key::String, M::SparseMatrixCSC) -> DiskCSCMatrix

Store the sparse matrix `M` under `key` and return its [`DiskCSCMatrix`](@ref) proxy. If `key` already exists, the existing proxy is returned unchanged. Matrices smaller than `cache.threshold` bytes are wrapped in an in-memory proxy (strongly held, never evicted) and are not written to disk.
"""
function store!(cache::DiskCache, key::String, M::SparseMatrixCSC{V, I})::DiskCSCMatrix{V, I} where {V, I}
    haskey(cache.entries, key) && return cache.entries[key]::DiskCSCMatrix{V, I}
    dm = if sizeof(M.colptr)+sizeof(M.rowval)+sizeof(M.nzval) >= cache.threshold
        dir = joinpath(cache.dir, string(cache.counter += 1, base=16))
        mkpath(dir)
        open(joinpath(dir, "header"), "w") do io
            print(io, string(V), " ", string(I), " ", M.m, " ", M.n, " ", nnz(M))
        end
        write(joinpath(dir, "colptr.bin"), M.colptr)
        write(joinpath(dir, "rowval.bin"), M.rowval)
        write(joinpath(dir, "nzval.bin"), M.nzval)
        DiskCSCMatrix{V, I}(dir, M.m, M.n, nnz(M), WeakRef(M))
    else
        DiskCSCMatrix{V, I}(nothing, M.m, M.n, nnz(M), M)
    end
    cache.entries[key] = dm
    return dm
end

"""
    empty!(cache::DiskCache) -> DiskCache

Remove all entries of `cache` and delete their files (best effort), extending `Base.empty!`. The directory-name counter is deliberately not reset, so that directories whose deletion failed (e.g. due to live memory mappings) are never reused for new contents.
"""
function Base.empty!(cache::DiskCache)
    for entry in values(cache.entries)
        isnothing(entry.path) && continue
        try rm(entry.path; recursive=true, force=true) catch end
    end
    empty!(cache.entries)
    return cache
end

const DISKCACHE = Ref{Union{Nothing, DiskCache}}(nothing)
"""
    diskcache() -> DiskCache

Get the process-wide default [`DiskCache`](@ref), creating it lazily on first use. Since the cache is keyed by serialized content, sharing it across different systems is safe: identical contents map to identical matrices, and distinct contents are never confused.
"""
@inline function diskcache()
    cache = DISKCACHE[]
    if isnothing(cache)
        cache = DiskCache()
        DISKCACHE[] = cache
    end
    return cache
end

"""
    diskcache!(cache::DiskCache) -> DiskCache

Set the process-wide default disk cache (e.g., to customize the parent directory of its files or the size threshold). The previous default is discarded (its directory is removed on a best-effort basis).
"""
function diskcache!(cache::DiskCache)
    isnothing(DISKCACHE[]) || empty!(DISKCACHE[])
    DISKCACHE[] = cache
    return cache
end
