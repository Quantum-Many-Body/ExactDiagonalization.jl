# Basics for exact diagonalization method
"""
    const edtimer = TimerOutput()

Default shared timer for all exact diagonalization methods.
"""
const edtimer = TimerOutput()

"""
    const notimer = TimerOutput()

Shared disabled timer for all exact diagonalization methods: the `@timeit` instrumentation against it is a no-op. It serves as the default timer of the parallel (`ntasks>1`) code paths, since `TimerOutput` is not thread-safe.
"""
const notimer = TimerOutput()
disable_timer!(notimer)

"""
    EDKind{K}

Kind of the exact diagonalization method applied to a quantum lattice system.
"""
struct EDKind{K} end
@inline EDKind(K::Symbol) = EDKind{K}()
@inline EDKind(object::Union{Hilbert, Internal, OperatorIndex}) = EDKind(typeof(object))
@inline EDKind(::Type{H}) where {H<:Hilbert} = EDKind(valtype(H))
@inline EDKind(::Type{I}) where {I<:Internal} = EDKind(eltype(I))
@inline EDKind(::Type{I}) where {I<:CompositeIndex} = EDKind(indextype(I))
@inline EDKind(::Type{I}) where {I<:Index} = EDKind(internalindextype(I))

"""
    Sector

A sector of the Hilbert space which forms the bases of an irreducible representation of the Hamiltonian of a quantum lattice system.
"""
abstract type Sector end
@inline Base.hash(sector::Sector, h::UInt) = hash(id(sector), h)
@inline Base.:(==)(sector₁::Sector, sector₂::Sector) = isequal(id(sector₁), id(sector₂))
@inline Base.isequal(sector₁::Sector, sector₂::Sector) = isequal(id(sector₁), id(sector₂))
@inline Base.show(io::IO, sector::Sector) = show(io, MIME"text/plain"(), sector)

"""
    match(sector₁::Sector, sector₂::Sector) -> Bool

Judge whether two sectors match each other, that is, whether they can be used together as the bra and ket spaces.
"""
@inline Base.match(::Sector, ::Sector) = false

"""
    sumable(sector₁::Sector, sector₂::Sector) -> Bool

Judge whether two sectors could be direct summed.
"""
@inline sumable(::Sector, ::Sector) = false

"""
    productable(sector₁::Sector, sector₂::Sector) -> Bool

Judge whether two sectors can form a direct product.
"""
@inline productable(::Sector, ::Sector) = false

"""
    broadcast(::Type{Sector}, quantumnumbers::OneAtLeast{Abelian}, hilbert::Hilbert, args...; table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))) -> NTuple{fieldcount(typeof(quantumnumbers)), Sector}

Construct a set of sectors based on the quantum numbers and a Hilbert space.
"""
@inline function Base.broadcast(::Type{Sector}, quantumnumbers::OneAtLeast{Abelian}, hilbert::Hilbert, args...; table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert)))
    return map(quantumnumbers) do quantumnumber
        Sector(quantumnumber, hilbert, args...; table=table)
    end
end

"""
    sparse(op::Operator{V, Tuple{}}, braket::NTuple{2, Sector}, table::AbstractDict, dtype=V) where V -> SparseMatrixCSC{dtype, Int}

Get the sparse matrix representation of a scalar operator between the bases of `braket`, extending [`SparseArrays.sparse`](https://sparsearrays.juliasparse.org/dev/#SparseArrays.sparse). For the memoized variant, see [`matrix`](@ref).
"""
@inline sparse(op::Operator{V, Tuple{}}, braket::NTuple{2, Sector}, ::AbstractDict, dtype=V) where V = sparse(one(dtype)*op.value*I, dimension(braket[1]), dimension(braket[2]))

"""
    matrix(op::OperatorIndex, braket::NTuple{2, Sector}, table::AbstractDict, dtype=Int; cached::Bool=false) -> DiskCSCMatrix{dtype, Int}

Get the sparse matrix representation of an operator index as a [`DiskCSCMatrix`](@ref) proxy. The keyword `cached` is forwarded to the memoization wrapper of [`matrix`](@ref).
"""
@inline matrix(op::OperatorIndex, braket::NTuple{2, Sector}, table::AbstractDict, dtype=Int; cached::Bool=false) = matrix(Operator(1, op), braket, table, dtype; cached=cached)

"""
    matrix(op::Operator, braket::NTuple{2, Sector}, table::AbstractDict, dtype=scalartype(op); cached::Bool=false) -> DiskCSCMatrix{dtype, Int}

Get the sparse matrix representation of an operator between the bases of `braket` as a [`DiskCSCMatrix`](@ref) proxy. With `cached=false` (the default), the proxy is anonymous: it is neither written to disk nor entered into the cache, and neither is anything derived from it by the algebra of `DiskCSCMatrix`. With `cached=true`, the matrix is stored in the session-wide [`DiskCache`](@ref) under its [`cachekey`](@ref) (on disk when larger than the threshold), and identical (operator, bases, table, element type) contents return the shared proxy without reconstructing. The construction is done by [`sparse`](@ref).

Here, `table` specifies the order of the operator indexes.
"""
function matrix(op::Operator, braket::NTuple{2, Sector}, table::AbstractDict, dtype=scalartype(op); cached::Bool=false)
    cached || return DiskCSCMatrix(sparse(op, braket, table, dtype))
    dm = get!(diskcache(), cachekey(op, braket, table, dtype)) do
        sparse(op, braket, table, dtype)
    end
    return dm::DiskCSCMatrix{dtype, Int}
end

"""
    matrix(ops::Operators, braket::NTuple{2, Sector}, table::AbstractDict, dtype=scalartype(ops); cached::Bool=false) -> DiskCSCMatrix{dtype, Int}

Get the sparse matrix representation of a set of operators between the bases of `braket` as a [`DiskCSCMatrix`](@ref) proxy (the sum of the per-operator proxies; see [`matrix`](@ref)). The recursive bisection construction is always anonymous: intermediate sums are neither cached nor written to disk. With `cached=true`, only the final block is memoized: its [`cachekey`](@ref) is looked up in the session-wide [`DiskCache`](@ref) before construction (a hit skips the construction entirely), and a miss is stored under that key afterwards. A length-1 set is delegated to the operator-level method so that identical contents share the same key.

Here, `table` specifies the order of the operator indexes.
"""
function matrix(ops::Operators, braket::NTuple{2, Sector}, table::AbstractDict, dtype=scalartype(ops); cached::Bool=false)
    length(ops)==0 && return DiskCSCMatrix(spzeros(dtype, length(braket[1]), length(braket[2])))
    length(ops)==1 && return matrix(ops[1], braket, table, dtype; cached=cached)
    cached || return matrix(ops[1:length(ops)÷2], braket, table, dtype) + matrix(ops[length(ops)÷2+1:length(ops)], braket, table, dtype)
    dm = get!(diskcache(), cachekey(ops, braket, table, dtype)) do
        sparse(matrix(ops, braket, table, dtype))
    end
    return dm::DiskCSCMatrix{dtype, Int}
end

"""
    cachekey(ops::Union{Operator, Operators}, braket::NTuple{2, Sector}, table::AbstractDict, dtype::Type{<:Number}) -> String

The content key of the matrix of an operator or a set of operators between the bases of `braket`, used by the memoized [`matrix`](@ref) wrapper: the serialized content of the operator(s) (including the coefficients), the defining content of the two bases (type, `id`, dimension), the index table, and the matrix element type. The key is the raw serialization bytes held in a `String` container (an opaque binary blob, not valid UTF-8 text; only `hash`/`isequal`/dictionary lookups are meaningful on it). Since the full content bytes are the key, distinct contents are never confused, regardless of hash collisions.

For an `Operators`, the key depends on the iteration order of the set (the insertion order of its `OrderedDict` contents): identical construction paths yield identical keys, while the same terms assembled in a different order yield a different key (a benign cache miss, never an error).
"""
function cachekey(ops::Union{Operator, Operators}, braket::NTuple{2, Sector}, table::AbstractDict, dtype::Type{<:Number})
    io = IOBuffer()
    serialize(io, (ops, map(bs->(typeof(bs), id(bs), dimension(bs)), braket), table, dtype))
    return String(take!(io))
end

"""
    EDMatrix{M<:DiskCSCMatrix, S<:Sector} <: OperatorPack{M, Tuple{S, S}}

Matrix representation of quantum operators between a ket Hilbert space and a bra Hilbert space.
"""
struct EDMatrix{M<:DiskCSCMatrix, S<:Sector} <: OperatorPack{M, Tuple{S, S}}
    matrix::M
    bra::S
    ket::S
end
@inline parameternames(::Type{<:EDMatrix}) = (:value, :sector)
@inline getcontent(m::EDMatrix, ::Val{:value}) = m.matrix
@inline getcontent(m::EDMatrix, ::Val{:id}) = (m.bra, m.ket)
@inline Base.promote_rule(M::Type{<:EDMatrix}, N::Type{<:Number}) = reparameter(M, :value, reparameter(valtype(M), 1, promote_type(scalartype(M), N)))
@inline showasleaf(::Type{<:EDMatrix}) = false
@inline Base.show(io::IO, m::EDMatrix) = show(io, MIME"text/plain"(), m)
@inline Base.show(io::IO, ::MIME"text/plain", m::EDMatrix) = showcontent(io, m)

"""
    EDMatrix(m::Union{DiskCSCMatrix, SparseMatrixCSC}, sector::Sector)
    EDMatrix(m::Union{DiskCSCMatrix, SparseMatrixCSC}, braket::NTuple{2, Sector})
    EDMatrix(m::Union{DiskCSCMatrix, SparseMatrixCSC}, bra::Sector, ket::Sector)

Construct a matrix representation when
1) the bra and ket Hilbert spaces share the same bases;
2) the bra and ket Hilbert spaces may be different;
3) the bra and ket Hilbert spaces may or may not be the same.

The sparse matrix payload is wrapped into an in-memory [`DiskCSCMatrix`](@ref) proxy.
"""
@inline EDMatrix(m::SparseMatrixCSC, sector::Sector) = EDMatrix(DiskCSCMatrix(m), sector)
@inline EDMatrix(m::SparseMatrixCSC, bra::Sector, ket::Sector) = EDMatrix(DiskCSCMatrix(m), bra, ket)
@inline EDMatrix(m::SparseMatrixCSC, braket::NTuple{2, Sector}) = EDMatrix(DiskCSCMatrix(m), braket)
@inline EDMatrix(m::DiskCSCMatrix, sector::Sector) = EDMatrix(m, sector, sector)
@inline EDMatrix(m::DiskCSCMatrix, braket::NTuple{2, Sector}) = EDMatrix(m, braket[1], braket[2])

"""
    EDEigenData{V<:Number, T<:Number, S<:Sector} <: Data

Eigen decomposition in exact diagonalization method.

Compared to the usual eigen decomposition `Eigen`, `EDEigenData` contains a `:sectors` attribute to store the sectors of Hilbert space in which the eigen values and eigen vectors are computed.
Furthermore, given that in different sectors the dimensions of the sub-Hilbert spaces can also be different, the `:vectors` attribute of `EDEigenData` is a vector of vector instead of a matrix.
"""
struct EDEigenData{V<:Number, T<:Number, S<:Sector} <: Data
    values::Vector{V}
    vectors::Vector{Vector{T}}
    sectors::Vector{S}
    function EDEigenData(values::AbstractVector{<:Number}, vectors::AbstractVector{<:AbstractVector{<:Number}}, sectors::AbstractVector{<:Sector})
        @assert length(values)==length(vectors)==length(sectors) "EDEigenData error: mismatched length of values, vectors and sectors."
        new{eltype(values), eltype(eltype(vectors)), eltype(sectors)}(values, vectors, sectors)
    end
end
@inline Base.iterate(data::EDEigenData) = (data.values, Val(:vectors))
@inline Base.iterate(data::EDEigenData, ::Val{:vectors}) = (data.vectors, Val(:sectors))
@inline Base.iterate(data::EDEigenData, ::Val{:sectors}) = (data.sectors, Val(:done))
@inline Base.iterate(data::EDEigenData, ::Val{:done}) = nothing

"""
    count(data::EDEigenData) -> Int

Count the number of eigen value-vector-sector groups contained in an `EDEigenData`.
"""
@inline Base.count(data::EDEigenData) = length(data.values)

"""
    eigen(m::EDMatrix; nev::Int=1, which::Symbol=:SR, tol::Real=1e-12, maxiter::Int=300, v₀::Union{AbstractVector{<:Number}, Int}=dimension(m.bra), krylovdim::Int=max(20, 2*nev+1), verbosity::Int=0) -> EDEigenData

Solve the eigen problem by the restarted Lanczos method provided by the [KrylovKit](https://github.com/Jutho/KrylovKit.jl) package.

When `v₀` is an integer, it is interpreted as the dimension of the Hilbert space and a random initial vector of eltype `scalartype(m)` is used; when `v₀` is a vector, it is used as the initial vector of the restarted Lanczos method and the eltype is inferred from `v₀` itself.
"""
@inline function eigen(m::EDMatrix; nev::Int=1, which::Symbol=:SR, tol::Real=1e-12, maxiter::Int=300, v₀::Union{AbstractVector{<:Number}, Int}=dimension(m.bra), krylovdim::Int=max(20, 2*nev+1), verbosity::Int=0)
    @assert m.bra==m.ket "eigen error: eigen decomposition of an `EDMatrix` are only available for those with the same bra and ket Hilbert spaces."
    if isa(v₀, Int)
        eigvals, eigvecs = eigsolve(sparse(m.matrix), v₀, nev, which, scalartype(m); krylovdim=krylovdim, tol=tol, maxiter=maxiter, ishermitian=true, verbosity=verbosity)
    else
        eigvals, eigvecs = eigsolve(sparse(m.matrix), v₀, nev, which; krylovdim=krylovdim, tol=tol, maxiter=maxiter, ishermitian=true, verbosity=verbosity)
    end
    if length(eigvals) > nev
        eigvals = eigvals[1:nev]
        eigvecs = eigvecs[1:nev]
    end
    return EDEigenData(eigvals, eigvecs, fill(m.bra, length(eigvals)))
end

"""
    eigen(
        ms::OperatorSum{<:EDMatrix};
        nev::Int=1,
        which::Symbol=:SR,
        tol::Real=1e-12,
        maxiter::Int=300,
        v₀::Union{Dict{<:Abelian, <:Union{AbstractVector{<:Number}, Int}}, Dict{<:Sector, <:Union{AbstractVector{<:Number}, Int}}}=Dict(m.ket=>dimension(m.ket) for m in ms),
        krylovdim::Int=max(20, 2*nev+1),
        verbosity::Int=0,
        timer::TimerOutput=edtimer
    ) -> EDEigenData

Solve the eigen problem by the restarted Lanczos method provided by the KrylovKit package.

The initial vectors/dimensions of the restarted Lanczos method can be assigned to the individual sectors by `v₀`, whose keys can be either the sectors or their corresponding Abelian quantum numbers, and whose values can be either vectors (used as the initial vectors) or integers (interpreted as the dimensions of the sectors, in which case random initial vectors are used). Sectors missing from `v₀` fall back to random initial vectors. An error is raised if any key of `v₀` matches no sector of `ms`.
"""
@inline function eigen(
    ms::OperatorSum{<:EDMatrix};
    nev::Int=1,
    which::Symbol=:SR,
    tol::Real=1e-12,
    maxiter::Int=300,
    v₀::Union{Dict{<:Abelian, <:Union{AbstractVector{<:Number}, Int}}, Dict{<:Sector, <:Union{AbstractVector{<:Number}, Int}}}=Dict(m.ket=>dimension(m.ket) for m in ms),
    krylovdim::Int=max(20, 2*nev+1),
    verbosity::Int=0,
    timer::TimerOutput=edtimer
)
    @timeit timer "eigen" begin
        values, vectors, sectors = real(scalartype(ms))[], Vector{scalartype(ms)}[], eltype(idtype(eltype(ms)))[]
        if isa(v₀, Dict{<:Abelian, <:Union{AbstractVector{<:Number}, Int}})
            unmatched = setdiff(keys(v₀), Set(Abelian(m.ket) for m in ms))
            isempty(unmatched) || error("eigen error: input v₀ contains Abelian quantum numbers ($(join(unmatched, ", "))) that match no sector.")
            v₀ = Dict(m.ket=>get(v₀, Abelian(m.ket), dimension(m.ket)) for m in ms)
        else
            unmatched = setdiff(keys(v₀), Set(m.ket for m in ms))
            isempty(unmatched) || error("eigen error: input v₀ contains sectors ($(join(unmatched, ", "))) that match no sector.")
        end
        for m in ms
            @timeit timer string(Abelian(m.ket)) begin
                k = min(dimension(m.ket), nev)
                eigensystem = eigen(m; nev=k, which=which, tol=tol, maxiter=maxiter, v₀=get(v₀, m.ket, dimension(m.ket)), krylovdim=krylovdim, verbosity=verbosity)
                for i = 1:k
                    push!(values, eigensystem.values[i])
                    push!(vectors, eigensystem.vectors[i])
                    push!(sectors, eigensystem.sectors[i])
                end
            end
        end
        nev>length(values) && @warn("Requested number ($nev) of eigen values exceeds the maximum available ($(length(values))).")
        perm = sortperm(values)[1:min(nev, length(values))]
        result = EDEigenData(values[perm], vectors[perm], sectors[perm])
    end
    return result
end

"""
    EDMatrixization{D<:Number, T<:AbstractDict, S<:Sector} <: Matrixization

Matrixization of a quantum lattice system on a target Hilbert space.
"""
struct EDMatrixization{D<:Number, T<:AbstractDict, S<:Sector} <: Matrixization
    table::T
    brakets::Vector{Tuple{S, S}}
    EDMatrixization{D}(table::AbstractDict, brakets::Vector{Tuple{S, S}}) where {D<:Number, S<:Sector} = new{D, typeof(table), S}(table, brakets)
end
@inline function Base.valtype(::Type{<:EDMatrixization{D, <:AbstractDict, S}}, ::Type{M}) where {D<:Number, S<:Sector, M<:Operator}
    @assert promote_type(D, valtype(M))==D "valtype error: convert $(valtype(M)) to $D is inexact."
    E = EDMatrix{DiskCSCMatrix{D, Int}, S}
    return OperatorSum{E, idtype(E)}
end
@inline Base.valtype(::Type{R}, ::Type{M}) where {R<:EDMatrixization, M<:OperatorSet} = valtype(R, eltype(M))
@inline contenttoconfig(matrixization::EDMatrixization) = (matrixization.table, [map(id, braket) for braket in matrixization.brakets])

"""
    (matrixization::EDMatrixization)(m::Union{Operator, Operators}; cached::Bool=true, kwargs...) -> OperatorSum{<:EDMatrix}

Get the matrix representation of an operator or a set of operators on the target Hilbert space as an `OperatorSum` of [`EDMatrix`](@ref) blocks, one per `braket` of `matrixization`. With `cached=true` (the default), each block is memoized in the session-wide [`DiskCache`](@ref) under its [`cachekey`](@ref) (see [`matrix`](@ref)), so that identical systems share the blocks without reconstructing them.
"""
function (matrixization::EDMatrixization)(m::Union{Operator, Operators}; cached::Bool=true, kwargs...)
    result = zero(valtype(matrixization, m))
    if isa(m, Operator) || length(m)>0
        for braket in matrixization.brakets
            add!(result, EDMatrix(matrix(m, braket, matrixization.table, scalartype(result); cached=cached), braket))
        end
    end
    return result
end

"""
    add!(destination, transformation::EDMatrixization, ops::OperatorSet; kwargs...)

Add the matrix representation of a whole set of operators to `destination`, specializing the generic element-wise contract of [`add!`](@ref) for [`EDMatrixization`](@ref): the set is matrixized as a whole (the result coincides by linearity), so that the memoization happens at the level of the merged block under the set-level [`cachekey`](@ref) instead of per operator.
"""
@inline function add!(destination, transformation::EDMatrixization, ops::OperatorSet; kwargs...)
    add!(destination, transformation(ops; kwargs...))
    return destination
end

"""
    EDMatrixization{D}(table::AbstractDict, sector::S, sectors::S...) where {D<:Number, S<:Sector}
    EDMatrixization{D}(table::AbstractDict, brakets::Vector{Tuple{S, S}}) where {D<:Number, S<:Sector}

Construct a matrixization.
"""
@inline EDMatrixization{D}(table::AbstractDict, sector::S, sectors::S...) where {D<:Number, S<:Sector} = EDMatrixization{D}(table, [(target, target) for target in (sector, sectors...)])

"""
    SectorFilter{S} <: LinearTransformation

Filter the target bra and ket Hilbert spaces.

It is a linear projection on `EDMatrix`: the matrix is kept if its id is contained in `brakets`, and zeroed otherwise. Since an `OperatorSum` never contains zero entries (zeros are always dropped by `add!`), applied to an `OperatorSum{<:EDMatrix}` the excluded entries are effectively removed from the result.
"""
struct SectorFilter{S} <: LinearTransformation
    brakets::Set{Tuple{S, S}}
end
@inline Base.valtype(::Type{<:SectorFilter}, ::Type{M}) where {M<:OperatorSet{<:EDMatrix}} = M
@inline (filter::SectorFilter)(m::EDMatrix) = id(m)∈filter.brakets ? m : EDMatrix(spzeros(scalartype(m), size(value(m))...), id(m))
@inline SectorFilter(sector::Sector, sectors::Sector...) = SectorFilter(map(target->(target, target), (sector, sectors...))...)
@inline SectorFilter(braket::NTuple{2, Sector}, brakets::NTuple{2, Sector}...) = SectorFilter(push!(Set{typeof(braket)}(), braket, brakets...))

"""
    ED{K<:EDKind, L<:Union{AbstractLattice, Nothing}, S<:Generator{<:Operators}, M<:EDMatrixization, H<:CategorizedGenerator{<:OperatorSum{<:EDMatrix}}} <: LatticeModel

Exact diagonalization method of a quantum lattice system.
"""
struct ED{K<:EDKind, L<:Union{AbstractLattice, Nothing}, S<:Generator{<:Operators}, M<:EDMatrixization, H<:CategorizedGenerator{<:OperatorSum{<:EDMatrix}}} <: LatticeModel
    lattice::L
    system::S
    matrixization::M
    H::H
    function ED{K}(lattice::Union{AbstractLattice, Nothing}, system::Generator{<:Operators}, matrixization::EDMatrixization) where {K<:EDKind}
        H = matrixization(empty(system))
        new{K, typeof(lattice), typeof(system), typeof(matrixization), typeof(H)}(lattice, system, matrixization, H)
    end
end
@inline Base.valtype(::Type{<:ED{<:EDKind, <:Union{AbstractLattice, Nothing}, <:Generator{<:Operators}, <:EDMatrixization, H}}) where {H<:CategorizedGenerator{<:OperatorSum{<:EDMatrix}}} = valtype(H)
@inline function update!(ed::ED; kwargs...)
    if length(kwargs)>0
        update!(ed.system; kwargs...)
        update!(ed.H, ed.matrixization, ed.system; kwargs...)
    end
    return ed
end
@inline Parameters(ed::ED) = Parameters(ed.system)
@inline contenttoconfig(ed::ED) = (ed.lattice, contenttoconfig(ed.system), contenttoconfig(ed.matrixization))

"""
    kind(ed::Union{ED, Algorithm{<:ED}}) -> EDKind
    kind(::Type{<:ED{K}}) where K -> EDKind
    kind(::Type{<:Algorithm{L}}) where {L<:ED} -> EDKind

Get the kind of an exact diagonalization method.
"""
@delegate @inline kind(ed::ED) = kind(typeof(ed))
@delegate @inline kind(::Type{<:ED{K}}) where K = K()

"""
    prepare!(ed::ED; timer::TimerOutput=edtimer) -> ED
    prepare!(ed::Algorithm{<:ED}; timer::TimerOutput=ed.timer) -> ED

Prepare the matrix representation.
"""
@delegate inject=(:timer,) @inline function prepare!(ed::ED; timer::TimerOutput=edtimer)
    isempty(ed.H) && @timeit timer "prepare!" begin
        reset!(ed.H, ed.matrixization, ed.system)
        @info "prepare! complete"
    end
    return ed
end

"""
    release!(ed::ED; gc::Bool=true) -> ED
    release!(ed::Algorithm{<:ED}; gc::Bool=true) -> ED

Release the memory source used in preparing the matrix representation. If `gc` is `true`, call the garbage collection immediately.
"""
@delegate @inline function release!(ed::ED; gc::Bool=true)
    empty!(ed.H)
    gc && GC.gc()
    return ed
end

"""
    matrix(ed::ED, sectors::Union{Abelian, Sector}...; timer::TimerOutput=edtimer, release::Bool=false) -> OperatorSum{<:EDMatrix}
    matrix(ed::Algorithm{<:ED}, sectors::Union{Abelian, Sector}...; timer::TimerOutput=ed.timer, release::Bool=false) -> OperatorSum{<:EDMatrix}

Get the sparse matrix representation of a quantum lattice system in the target space.
"""
@delegate inject=(:timer,) function matrix(ed::ED; timer::TimerOutput=edtimer, release::Bool=false)
    prepare!(ed; timer=timer)
    @timeit timer "matrix" begin
        result = expand(ed.H)
    end
    release && release!(ed; gc=true)
    return result
end
@delegate inject=(:timer,) function matrix(ed::ED, sector::Sector, sectors::Sector...; timer::TimerOutput=edtimer, release::Bool=false)
    prepare!(ed; timer=timer)
    @timeit timer "matrix" begin
        result = expand(SectorFilter(sector, sectors...)(ed.H))
    end
    release && release!(ed; gc=true)
    return result
end
@delegate inject=(:timer,) function matrix(ed::ED, quantumnumber::Abelian, quantumnumbers::Abelian...; timer::TimerOutput=edtimer, release::Bool=false)
    sectors = eltype(eltype(ed.matrixization.brakets))[]
    for (bra, ket) in ed.matrixization.brakets
        @assert bra==ket "matrix error: unequal bra and ket Hilbert spaces found."
        Abelian(bra)∈(quantumnumber, quantumnumbers...) && push!(sectors, bra)
    end
    return matrix(ed, sectors...; timer=timer, release=release)
end

"""
    sparse(ed::ED, sector::Sector; timer::TimerOutput=edtimer) -> SparseMatrixCSC

Get the sparse matrix representation of a quantum lattice system in a single sector, extending [`SparseArrays.sparse`](https://sparsearrays.juliasparse.org/dev/#SparseArrays.sparse). The construction reuses the cached operator blocks from the session-wide [`DiskCache`](@ref) implicitly.
"""
function sparse(ed::ED, sector::Sector; timer::TimerOutput=edtimer)
    m = if (sector, sector) ∈ ed.matrixization.brakets
        matrix(ed, sector; timer=timer)
    else
        @timeit timer "matrix" EDMatrixization{scalartype(ed)}(ed.matrixization.table, sector)(expand(ed.system))
    end
    return sparse(value(only(m)))
end

"""
    eigen(ed::ED, sectors::Union{Abelian, Sector}...; timer::TimerOutput=edtimer, release::Bool=false, kwargs...) -> EDEigenData
    eigen(ed::Algorithm{<:ED}, sectors::Union{Abelian, Sector}...; timer::TimerOutput=ed.timer, release::Bool=false, kwargs...) -> EDEigenData

Solve the eigen problem by the restarted Lanczos method provided by the KrylovKit package.
"""
@delegate inject=(:timer,) @inline eigen(ed::ED, sectors::Union{Abelian, Sector}...; timer::TimerOutput=edtimer, release::Bool=false, kwargs...) = eigen(matrix(ed, sectors...; timer=timer, release=release); timer=timer, kwargs...)

"""
    ED(system::OperatorGenerator, dtype::Type{<:Number}=scalartype(system))
    ED(lattice::Union{AbstractLattice, Nothing}, system::OperatorGenerator, dtype::Type{<:Number}=scalartype(system))
    ED(system::OperatorGenerator, quantumnumbers::OneOrMore{Abelian}, dtype::Type{<:Number}=scalartype(system))
    ED(lattice::Union{AbstractLattice, Nothing}, system::OperatorGenerator, quantumnumbers::OneOrMore{Abelian}, dtype::Type{<:Number}=scalartype(system))

Construct the exact diagonalization method from a pre-built `OperatorGenerator` with/without quantum numbers.
"""
@inline ED(system::OperatorGenerator, dtype::Type{<:Number}=scalartype(system)) = ED(nothing, system, dtype)
function ED(lattice::Union{AbstractLattice, Nothing}, system::OperatorGenerator, dtype::Type{<:Number}=scalartype(system))
    hilbert = system.hilbert
    kind = typeof(EDKind(hilbert))
    table = Table(hilbert, Metric(EDKind(hilbert), hilbert))
    matrixization = EDMatrixization{dtype}(table, Sector(hilbert))
    return ED{kind}(lattice, system, matrixization)
end
@inline ED(system::OperatorGenerator, quantumnumbers::OneOrMore{Abelian}, dtype::Type{<:Number}=scalartype(system)) = ED(nothing, system, quantumnumbers, dtype)
function ED(lattice::Union{AbstractLattice, Nothing}, system::OperatorGenerator, quantumnumbers::OneOrMore{Abelian}, dtype::Type{<:Number}=scalartype(system))
    hilbert = system.hilbert
    kind = typeof(EDKind(hilbert))
    table = Table(hilbert, Metric(EDKind(hilbert), hilbert))
    sectors = broadcast(Sector, OneOrMore(quantumnumbers), hilbert; table=table)
    matrixization = EDMatrixization{dtype}(table, sectors...)
    return ED{kind}(lattice, system, matrixization)
end

"""
    ED(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms)
    )
    ED(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms)
    )

Construct the exact diagonalization method for a quantum lattice system with/without quantum numbers.
"""
@inline function ED(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms)
)
    system = Generator(bonds(lattice, neighbors), hilbert, normalize(terms), boundary; half=false)
    return ED(lattice, system, dtype)
end
@inline function ED(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms)
)
    system = Generator(bonds(lattice, neighbors), hilbert, normalize(terms), boundary; half=false)
    return ED(lattice, system, quantumnumbers, dtype)
end

"""
    normalize(term::Term) -> Term
    normalize(terms::OneAtLeast{Term}) -> OneAtLeast{Term}

Normalize the terms for exact diagonalization.

For a `Pairing` term, its factor is multiplied by `1//2` because in contrast to the BdG-form convention, the exact diagonalization method does not need to symmetrize the pairing operators, so the coefficient should not be doubled.
"""
@inline normalize(term::Term) = term
@inline normalize(term::Pairing) = replace(term; factor=1//2)
@inline normalize(terms::OneAtLeast{Term}) = map(normalize, terms)
