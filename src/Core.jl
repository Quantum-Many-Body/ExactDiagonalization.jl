using Base.Iterators: product
using KrylovKit: eigsolve
using LinearAlgebra: I, dot
using LuxurySparse: SparseMatrixCOO
using Printf: @printf
using QuantumLattices: eager, plain, bonds, decompose, expand, findindex, idtype, indextype, internalindextype, iscreation, nneighbor, reparameter, reset!, shape, statistics, totalspin, value
using QuantumLattices: Abelian, AbstractLattice, Action, Algorithm, Assignment, Boundary, BrillouinZone, CategorizedGenerator, Combinations, CompositeIndex, Data, DuplicatePermutations, Fock, FockIndex, Frontend, Generator, Hilbert, Index, Internal, InternalIndex, LinearTransformation, Matrixization, Metric, Neighbors, OneAtLeast, OneOrMore, Operator, OperatorIndex, OperatorIndexToTuple, OperatorPack, Operators, OperatorSum, ReciprocalSpace, ReciprocalZone, Spin, SpinIndex, Table, Term, VectorSpace, VectorSpaceEnumerative, VectorSpaceStyle, ℕ, 𝕊, 𝕊ᶻ, ℤ₁
using SparseArrays: SparseMatrixCSC, nnz, nonzeros, nzrange, rowvals, sparse, spzeros
using TimerOutputs: TimerOutput, @timeit

import LinearAlgebra: eigen
import QuantumLattices: Graded, Parameters, ⊠, ⊗, ⊕, add!, dimension, getcontent, id, kind, matrix, options, parameternames, partition, run!, scalartype, update!

# Basics for exact diagonalization method
"""
    const edtimer = TimerOutput()

Default shared timer for all exact diagonalization methods.
"""
const edtimer = TimerOutput()

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

"""
    match(sector₁::Sector, sector₂::Sector) -> Bool

Judge whether two sectors match each other, that is, whether they can be used together as the bra and ket spaces.
"""
@inline Base.match(sector₁::Sector, sector₂::Sector) = false

"""
    sumable(sector₁::Sector, sector₂::Sector) -> Bool

Judge whether two sectors could be direct summed.
"""
@inline sumable(::Sector, ::Sector) = false

"""
    productable(sector₁::Sector, sector₂::Sector) -> Bool

Judge whether two sectors could be direct producted.
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
    matrix(ops::Operators, braket::NTuple{2, Sector}, table::AbstractDict, dtype=scalartype(ops)) -> SparseMatrixCSC{dtype, Int}

Get the CSC-formed sparse matrix representation of a set of operators.

Here, `table` specifies the order of the operator indexes.
"""
function matrix(ops::Operators, braket::NTuple{2, Sector}, table::AbstractDict, dtype=scalartype(ops))
    length(ops)==0 && return spzeros(dtype, length(braket[1]), length(braket[2]))
    length(ops)==1 && return matrix(ops[1], braket, table, dtype)
    return matrix(ops[1:length(ops)÷2], braket, table, dtype) + matrix(ops[length(ops)÷2+1:length(ops)], braket, table, dtype)
end

"""
    EDMatrix{M<:SparseMatrixCSC, S<:Sector} <: OperatorPack{M, Tuple{S, S}}

Matrix representation of quantum operators between a ket Hilbert space and a bra Hilbert space.
"""
struct EDMatrix{M<:SparseMatrixCSC, S<:Sector} <: OperatorPack{M, Tuple{S, S}}
    matrix::M
    bra::S
    ket::S
end
@inline parameternames(::Type{<:EDMatrix}) = (:value, :sector)
@inline getcontent(m::EDMatrix, ::Val{:value}) = m.matrix
@inline getcontent(m::EDMatrix, ::Val{:id}) = (m.bra, m.ket)
@inline Base.promote_rule(M::Type{<:EDMatrix}, N::Type{<:Number}) = reparameter(M, :value, reparameter(valtype(M), 1, promote_type(scalartype(M), N)))

"""
    EDMatrix(m::SparseMatrixCSC, sector::Sector)
    EDMatrix(m::SparseMatrixCSC, braket::NTuple{2, Sector})
    EDMatrix(m::SparseMatrixCSC, bra::Sector, ket::Sector)

Construct a matrix representation when
1) the bra and ket Hilbert spaces share the same bases;
2) the bra and ket Hilbert spaces may be different;
3) the bra and ket Hilbert spaces may or may not be the same.
"""
@inline EDMatrix(m::SparseMatrixCSC, sector::Sector) = EDMatrix(m, sector, sector)
@inline EDMatrix(m::SparseMatrixCSC, braket::NTuple{2, Sector}) = EDMatrix(m, braket[1], braket[2])

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
"""
@inline function eigen(m::EDMatrix; nev::Int=1, which::Symbol=:SR, tol::Real=1e-12, maxiter::Int=300, v₀::Union{AbstractVector{<:Number}, Int}=dimension(m.bra), krylovdim::Int=max(20, 2*nev+1), verbosity::Int=0)
    @assert m.bra==m.ket "eigen error: eigen decomposition of an `EDMatrix` are only available for those with the same bra and ket Hilbert spaces."
    eigvals, eigvecs = eigsolve(m.matrix, v₀, nev, which, scalartype(m); krylovdim=krylovdim, tol=tol, maxiter=maxiter, ishermitian=true, verbosity=verbosity)
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
        v₀::Union{Dict{<:Abelian, <:Union{AbstractVector{<:Number}, Int}}, Dict{<:Sector, <:Union{AbstractVector{<:Number}, Int}}}=Dict(Abelian(m.ket)=>dimension(m.ket) for m in ms),
        krylovdim::Int=max(20, 2*nev+1),
        verbosity::Int=0,
        timer::TimerOutput=edtimer
    ) -> EDEigenData

Solve the eigen problem by the restarted Lanczos method provided by the Arpack package.
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
        isa(v₀, Dict{<:Abelian, <:Union{AbstractVector{<:Number}, Int}}) && (v₀ = Dict(m.ket=>get(v₀, Abelian(m.ket), dimension(m.ket)) for m in ms))
        for m in ms
            @timeit timer string(Abelian(m.ket)) begin
                k = min(dimension(m.ket), nev)
                eigensystem = eigen(m; nev=k, which=which, tol=tol, maxiter=maxiter, v₀=get(v₀, Abelian(m.ket), dimension(m.ket)), krylovdim=krylovdim, verbosity=verbosity)
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
    E = EDMatrix{SparseMatrixCSC{D, Int}, S}
    return OperatorSum{E, idtype(E)}
end
@inline Base.valtype(::Type{R}, ::Type{M}) where {R<:EDMatrixization, M<:Operators} = valtype(R, eltype(M))
function (matrixization::EDMatrixization)(m::Union{Operator, Operators}; kwargs...)
    result = zero(valtype(matrixization, m))
    if isa(m, Operator) || length(m)>0
        for braket in matrixization.brakets
            add!(result, EDMatrix(matrix(m, braket, matrixization.table, scalartype(result); kwargs...), braket))
        end
    end
    return result
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
"""
struct SectorFilter{S} <: LinearTransformation
    brakets::Set{Tuple{S, S}}
end
@inline Base.valtype(::Type{<:SectorFilter}, ::Type{M}) where {M<:OperatorSum{<:EDMatrix}} = M
@inline (sectorfileter::SectorFilter)(m::EDMatrix) = id(m)∈sectorfileter.brakets ? m : EDMatrix(spzeros(scalartype(m), size(value(m))...), id(m))
@inline SectorFilter(sector::Sector, sectors::Sector...) = SectorFilter(map(target->(target, target), (sector, sectors...))...)
@inline SectorFilter(braket::NTuple{2, Sector}, brakets::NTuple{2, Sector}...) = SectorFilter(push!(Set{typeof(braket)}(), braket, brakets...))

"""
    ED{K<:EDKind, L<:Union{AbstractLattice, Nothing}, S<:Generator{<:Operators}, M<:EDMatrixization, H<:CategorizedGenerator{<:OperatorSum{<:EDMatrix}}} <: Frontend

Exact diagonalization method of a quantum lattice system.
"""
struct ED{K<:EDKind, L<:Union{AbstractLattice, Nothing}, S<:Generator{<:Operators}, M<:EDMatrixization, H<:CategorizedGenerator{<:OperatorSum{<:EDMatrix}}} <: Frontend
    lattice::L
    system::S
    matrixization::M
    H::H
    function ED{K}(lattice::Union{AbstractLattice, Nothing}, system::Generator{<:Operators}, matrixization::EDMatrixization) where {K<:EDKind}
        H = matrixization(empty(system))
        new{K, typeof(lattice), typeof(system), typeof(matrixization), typeof(H)}(lattice, system, matrixization, H)
    end
end
@inline kind(ed::ED) = kind(typeof(ed))
@inline kind(::Type{<:ED{K}}) where K = K()
@inline scalartype(::Type{<:ED{<:EDKind, <:AbstractLattice, S}}) where {S<:Generator{<:Operators}} = scalartype(S)
@inline scalartype(::Type{<:Algorithm{F}}) where {F<:ED} = scalartype(F)
@inline function update!(ed::ED; kwargs...)
    if length(kwargs)>0
        update!(ed.system; kwargs...)
        update!(ed.H, ed.matrixization, ed.system; kwargs...)
    end
    return ed
end
@inline Parameters(ed::ED) = Parameters(ed.system)

"""
    prepare!(ed::ED; timer::TimerOutput=edtimer) -> ED
    prepare!(ed::Algorithm{<:ED}) -> Algorithm{<:ED}

Prepare the matrix representation.
"""
@inline function prepare!(ed::ED; timer::TimerOutput=edtimer)
    isempty(ed.H) && @timeit timer "prepare!" begin
        reset!(ed.H, ed.matrixization, ed.system)
        @info "ED prepare complete."
    end
    return ed
end
@inline prepare!(ed::Algorithm{<:ED}) = (prepare!(ed.frontend; timer=ed.timer); ed)

"""
    release!(ed::ED; gc::Bool=true) -> ED
    release!(ed::Algorithm{<:ED}; gc::Bool=true) -> Algorithm{<:ED}

Release the memory source used in preparing the matrix representation. If `gc` is `true`, call the garbage collection immediately.
"""
@inline function release!(ed::ED; gc::Bool=true)
    empty!(ed.H)
    gc && GC.gc()
    return ed
end
@inline release!(ed::Algorithm{<:ED}; gc::Bool=true) = release!(ed.frontend; gc=gc)

"""
    matrix(ed::ED, sectors::Union{Abelian, Sector}...; timer::TimerOutput=edtimer, release::Bool=false, kwargs...) -> OperatorSum{<:EDMatrix}
    matrix(ed::Algorithm{<:ED}, sectors::Union{Abelian, Sector}...; release::Bool=false, kwargs...) -> OperatorSum{<:EDMatrix}

Get the sparse matrix representation of a quantum lattice system in the target space.
"""
function matrix(ed::ED; timer::TimerOutput=edtimer, release::Bool=false, kwargs...)
    prepare!(ed; timer=timer)
    @timeit timer "matrix" begin
        result = expand(ed.H)
    end
    release && release!(ed; gc=true)
    return result
end
function matrix(ed::ED, sector::Sector, sectors::Sector...; timer::TimerOutput=edtimer, release::Bool=false, kwargs...)
    prepare!(ed; timer=timer)
    @timeit timer "matrix" begin
        result = expand(SectorFilter(sector, sectors...)(ed.H))
    end
    release && release!(ed; gc=true)
    return result
end
function matrix(ed::ED, quantumnumber::Abelian, quantumnumbers::Abelian...; timer::TimerOutput=edtimer, kwargs...)
    sectors = eltype(eltype(ed.matrixization.brakets))[]
    for (bra, ket) in ed.matrixization.brakets
        @assert bra==ket "matrix error: unequal bra and ket Hilbert spaces found."
        Abelian(bra)∈(quantumnumber, quantumnumbers...) && push!(sectors, bra)
    end
    return matrix(ed, sectors...; timer=timer, kwargs...)
end
function matrix(ed::Algorithm{<:ED}, sectors::Union{Abelian, Sector}...; kwargs...)
    return matrix(ed.frontend, sectors...; timer=ed.timer, kwargs...)
end

"""
    eigen(ed::ED, sectors::Union{Abelian, Sector}...; timer::TimerOutput=edtimer, release::Bool=false, kwargs...) -> EDEigenData
    eigen(ed::Algorithm{<:ED}, sectors::Union{Abelian, Sector}...; release::Bool=false, kwargs...) -> EDEigenData

Solve the eigen problem by the restarted Lanczos method provided by the Arpack package.
"""
@inline eigen(ed::ED, sectors::Union{Abelian, Sector}...; timer::TimerOutput=edtimer, release::Bool=false, kwargs...) = eigen(matrix(ed, sectors...; timer=timer, release=release); timer=timer, kwargs...)
@inline eigen(ed::Algorithm{<:ED}, sectors::Union{Abelian, Sector}...; release::Bool=false, kwargs...) = eigen(ed.frontend, sectors...; timer=ed.timer, release=release, kwargs...)

"""
    ED(system::Generator{<:Operators}, table::AbstractDict, sectors::OneOrMore{Sector}, dtype::Type{<:Number}=scalartype(system))
    ED(lattice::Union{AbstractLattice, Nothing}, system::Generator{<:Operators}, table::AbstractDict, sectors::OneOrMore{Sector}, dtype::Type{<:Number}=scalartype(system))

Construct the exact diagonalization method for a quantum lattice system.
"""
@inline ED(system::Generator{<:Operators}, table::AbstractDict, sectors::OneOrMore{Sector}, dtype::Type{<:Number}=scalartype(system)) = ED(nothing, system, table, sectors, dtype)
@inline function ED(lattice::Union{AbstractLattice, Nothing}, system::Generator{<:Operators}, table::AbstractDict, sectors::OneOrMore{Sector}, dtype::Type{<:Number}=scalartype(system))
    kind = typeof(EDKind(eltype(eltype(system))))
    matrixization = EDMatrixization{dtype}(table, sectors)
    return ED{kind}(lattice, system, matrixization)
end

"""
    ED(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms)
    )
    ED(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, table::AbstractDict, sectors::OneOrMore{Sector}=Sector(hilbert; table=table), boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms)
    )

Construct the exact diagonalization method for a quantum lattice system.
"""
@inline function ED(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms)
)
    return ED(lattice, hilbert, terms, Table(hilbert, Metric(EDKind(hilbert), hilbert)), Sector(hilbert), boundary, dtype; neighbors=neighbors)
end
function ED(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, table::AbstractDict, sectors::OneOrMore{Sector}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms)
)
    system = Generator(bonds(lattice, neighbors), hilbert, OneOrMore(terms), boundary, eager; half=false)
    matrixization = EDMatrixization{dtype}(table, OneOrMore(sectors)...)
    return ED{typeof(EDKind(hilbert))}(lattice, system, matrixization)
end

"""
    ED(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms)
    )
    ED(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, table::AbstractDict, quantumnumbers::OneOrMore{Abelian}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms)
    )

Construct the exact diagonalization method for a quantum lattice system.
"""
@inline function ED(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms)
)
    return ED(lattice, hilbert, terms, Table(hilbert, Metric(EDKind(hilbert), hilbert)), quantumnumbers, boundary, dtype; neighbors=neighbors)
end
@inline function ED(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, table::AbstractDict, quantumnumbers::OneOrMore{Abelian}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms)
)
    return ED(lattice, hilbert, terms, table, broadcast(Sector, OneOrMore(quantumnumbers), hilbert; table=table), boundary, dtype; neighbors=neighbors)
end

"""
    const basicoptions = (
        nev = "number of eigenvalues to be computed",
        which = "type of eigenvalues to be computed",
        tol = "tolerance of the computation",
        maxiter = "maximum iteration of the computation",
        v₀ = "initial state",
        krylovdim = "maximum dimension of the Krylov subspace that will be constructed",
        verbosity = "verbosity level"
    )

Basic options of actions for exact diagonalization method.
"""
const basicoptions = (
    nev = "number of eigenvalues to be computed",
    which = "type of eigenvalues to be computed",
    tol = "tolerance of the computation",
    maxiter = "maximum iteration of the computation",
    v₀ = "initial state",
    krylovdim = "maximum dimension of the Krylov subspace that will be constructed",
    verbosity = "verbosity level",
    release = "release or not the cache for the construction of the sparse matrix representation of the Hamiltonian"
)

"""
    EDEigen{S<:Tuple{Vararg{Union{Abelian, Sector}}}} <: Action

Eigen system by exact diagonalization method.
"""
struct EDEigen{S<:Tuple{Vararg{Union{Abelian, Sector}}}} <: Action
    sectors::S
end
@inline options(::Type{<:Assignment{<:EDEigen}}) = basicoptions

"""
    EDEigen(sectors::Union{Abelian, Sector}...)
    EDEigen(sectors:::Tuple{Vararg{Union{Abelian, Sector}}})

Construct an `EDEigen`.
"""
@inline EDEigen(sectors::Union{Abelian, Sector}...) = EDEigen(sectors)
@inline run!(ed::Algorithm{<:ED}, edeigen::Assignment{<:EDEigen}; options...) = eigen(ed, edeigen.action.sectors...; options...)

# Binary bases, used for canonical fermionic and hard-core bosonic systems by default
## Basics for binary bases
"""
    basistype(i::Integer)
    basistype(::Type{I}) where {I<:Integer}

Get the binary basis type corresponding to an integer or a type of an integer.
"""
@inline basistype(i::Integer) = basistype(typeof(i))
@inline basistype(::Type{T}) where {T<:Unsigned} = T
@inline basistype(::Type{Int8}) = UInt8
@inline basistype(::Type{Int16}) = UInt16
@inline basistype(::Type{Int32}) = UInt32
@inline basistype(::Type{Int64}) = UInt64
@inline basistype(::Type{Int128}) = UInt128

"""
    BinaryBasis{I<:Unsigned}

Binary basis represented by an unsigned integer.

Here, we adopt the following common rules:
1. In a binary basis, a bit of an unsigned integer represents a single-particle state that can be occupied (`1`) or unoccupied (`0`).
2. The position of this bit in the unsigned integer counting from the right corresponds to the sequence of the single-particle state specified by a table.
3. When representing a many-body state by creation operators, they are arranged in ascending order according to their sequences.

In this way, any many-body state of canonical fermionic or hardcore bosonic systems can be represented ambiguously by the binary bases, e.g., ``c^†_2c^†_3c^†_4|\\text{Vacuum}\\rangle`` is represented by ``1110``.
"""
struct BinaryBasis{I<:Unsigned}
    rep::I
    BinaryBasis{I}(i::Integer) where {I<:Unsigned} = new{I}(convert(I, i))
end
@inline BinaryBasis(i::Integer) = (rep = Unsigned(i); BinaryBasis{typeof(rep)}(rep))
@inline Base.:(==)(basis₁::BinaryBasis, basis₂::BinaryBasis) = basis₁.rep == basis₂.rep
@inline Base.isequal(basis₁::BinaryBasis, basis₂::BinaryBasis) = isequal(basis₁.rep, basis₂.rep)
@inline Base.:<(basis₁::BinaryBasis, basis₂::BinaryBasis) = basis₁.rep < basis₂.rep
@inline Base.isless(basis₁::BinaryBasis, basis₂::BinaryBasis) = isless(basis₁.rep, basis₂.rep)
@inline Base.one(basis::BinaryBasis) = one(typeof(basis))
@inline Base.one(::Type{BinaryBasis{I}}) where {I<:Unsigned} = BinaryBasis(one(I))
@inline Base.zero(basis::BinaryBasis) = zero(typeof(basis))
@inline Base.zero(::Type{BinaryBasis{I}}) where {I<:Unsigned} = BinaryBasis(zero(I))
@inline Base.show(io::IO, basis::BinaryBasis) = @printf io "%s" string(basis.rep, base=2)
@inline Base.eltype(basis::BinaryBasis) = eltype(typeof(basis))
@inline Base.eltype(::Type{<:BinaryBasis}) = Int
@inline Base.IteratorSize(::Type{<:BinaryBasis}) = Base.SizeUnknown()

"""
    BinaryBasis(states; filter=index->true)
    BinaryBasis{I}(states; filter=index->true) where {I<:Unsigned}

Construct a binary basis with the given occupied states.
"""
@inline BinaryBasis(states; filter=index->true) = BinaryBasis{basistype(eltype(states))}(states; filter=filter)
function BinaryBasis{I}(states; filter=index->true) where {I<:Unsigned}
    rep, eye = zero(I), one(I)
    for (index, state) in enumerate(states)
        filter(index) && (rep += eye<<(state-1))
    end
    return BinaryBasis(rep)
end

"""
    iterate(basis::BinaryBasis)
    iterate(basis::BinaryBasis, state)

Iterate over the numbers of the occupied single-particle states.
"""
function Base.iterate(basis::BinaryBasis, state=(0, basis.rep))
    pos, rep = state
    while rep>0
        pos += 1
        isodd(rep) && return (pos, (pos, rep÷2))
        rep ÷= 2
    end
    return nothing
end

"""
    one(basis::BinaryBasis, state::Integer) -> BinaryBasis

Get a new binary basis with the specified single-particle state occupied.
"""
@inline Base.one(basis::BinaryBasis, state::Integer) = BinaryBasis(basis.rep | one(basis.rep)<<(state-1))

"""
    isone(basis::BinaryBasis, state::Integer) -> Bool

Judge whether the specified single-particle state is occupied for a binary basis.
"""
@inline Base.isone(basis::BinaryBasis, state::Integer) = (basis.rep & one(basis.rep)<<(state-1)) > 0

"""
    zero(basis::BinaryBasis, state::Integer) -> BinaryBasis

Get a new binary basis with the specified single-particle state unoccupied.
"""
@inline Base.zero(basis::BinaryBasis, state::Integer) = BinaryBasis(basis.rep & ~(one(basis.rep)<<(state-1)))

"""
    iszero(basis::BinaryBasis, state::Integer) -> Bool

Judge whether the specified single-particle state is unoccupied for a binary basis.
"""
@inline Base.iszero(basis::BinaryBasis, state::Integer) = !isone(basis, state)

"""
    count(basis::BinaryBasis) -> Int
    count(basis::BinaryBasis, start::Integer, stop::Integer) -> Int

Count the number of occupied single-particle states for a binary basis.
"""
@inline Base.count(basis::BinaryBasis) = count(basis, 1, ndigits(basis.rep, base=2))
@inline function Base.count(basis::BinaryBasis, start::Integer, stop::Integer)
    result = 0
    for i = start:stop
        isone(basis, i) && (result += 1)
    end
    return result
end

"""
    ⊗(basis₁::BinaryBasis, basis₂::BinaryBasis) -> BinaryBasis

Get the direct product of two binary bases.
"""
@inline ⊗(basis₁::BinaryBasis, basis₂::BinaryBasis) = BinaryBasis(basis₁.rep | basis₂.rep)

"""
    BinaryBasisRange{I<:Unsigned} <: VectorSpace{BinaryBasis{I}}

A continuous range of binary basis from 0 to 2^n-1.
"""
struct BinaryBasisRange{I<:Unsigned} <: VectorSpace{BinaryBasis{I}}
    slice::UnitRange{I}
    BinaryBasisRange(nstate::Integer) = new{basistype(nstate)}(zero(basistype(nstate)):basistype(nstate)(2^nstate-1))
end
@inline Base.issorted(::BinaryBasisRange) = true
@inline Base.length(bbr::BinaryBasisRange) = length(bbr.slice)
@inline Base.getindex(bbr::BinaryBasisRange, i::Integer) = BinaryBasis(bbr.slice[i])

"""
    BinaryBases{A<:Abelian, B<:BinaryBasis, T<:AbstractVector{B}} <: Sector

A set of binary bases.
"""
struct BinaryBases{A<:Abelian, B<:BinaryBasis, T<:AbstractVector{B}} <: Sector
    quantumnumbers::Vector{A}
    stategroups::Vector{B}
    table::T
end
@inline id(bs::BinaryBases) = (bs.quantumnumbers, bs.stategroups)
@inline dimension(bs::BinaryBases) = length(bs)
@inline Base.length(bs::BinaryBases) = length(bs.table)
@inline Base.firstindex(bs::BinaryBases) = firstindex(bs.table)
@inline Base.lastindex(bs::BinaryBases) = lastindex(bs.table)
@inline Base.getindex(bs::BinaryBases, i::Integer) = bs.table[i]
@inline Base.eltype(bs::BinaryBases) = eltype(typeof(bs))
@inline Base.eltype(::Type{<:BinaryBases{<:Abelian, B}}) where {B<:BinaryBasis} = B
@inline Base.iterate(bs::BinaryBases, state=1) = state>length(bs) ? nothing : (bs.table[state], state+1)
function Base.show(io::IO, bs::BinaryBases)
    for (i, (qn, group)) in enumerate(zip(bs.quantumnumbers, bs.stategroups))
        @printf io "{2^[%s]: %s}" join(collect(group), " ") qn
        i<length(bs.quantumnumbers) && @printf io "%s" " ⊗ "
    end
end
@inline Base.searchsortedfirst(b::BinaryBasis, bs::BinaryBases) = searchsortedfirst(bs.table, b)
@inline Base.searchsortedfirst(b::BinaryBasis, ::BinaryBases{<:Abelian, <:BinaryBasis, <:BinaryBasisRange}) = Int(b.rep+1)
@inline Base.match(bs₁::BinaryBases{A}, bs₂::BinaryBases{A}) where {A<:Abelian} = bs₁.stategroups == bs₂.stategroups
function sumable(bs₁::BinaryBases{A}, bs₂::BinaryBases{A}) where {A<:Abelian}
    Abelian(bs₁)==Abelian(bs₂) || return true
    productable(bs₁, bs₂) && return true
    return length(intersect(bs₁, bs₂))==0
end
function productable(bs₁::BinaryBases{A}, bs₂::BinaryBases{A}) where {A<:Abelian}
    for (group₁, group₂) in product(bs₁.stategroups, bs₂.stategroups)
        isequal(group₁.rep & group₂.rep, 0) || return false
    end
    return true
end

"""
    Abelian(bs::BinaryBases)

Get the Abelian quantum number of a set of binary bases.
"""
@inline Abelian(bs::BinaryBases) = sum(bs.quantumnumbers)

"""
    BinaryBases(states)
    BinaryBases(nstate::Integer)

Construct a set of binary bases that subjects to no quantum number conservation.
"""
function BinaryBases(nstate::Integer)
    stategroup = BinaryBasis(one(nstate):nstate)
    table = BinaryBasisRange(nstate)
    return BinaryBases([ℤ₁(0)], [stategroup], table)
end
function BinaryBases(states)
    stategroup = BinaryBasis(states)
    table = BinaryBasis{basistype(eltype(states))}[]
    table!(table, NTuple{length(states), basistype(eltype(states))}(sort!(collect(states); rev=true)))
    return BinaryBases([ℤ₁(0)], [stategroup], table)
end
function table!(table, states::NTuple{N}) where N
    for poses in DuplicatePermutations{N}((false, true))
        push!(table, BinaryBasis(states; filter=index->poses[index]))
    end
    return table
end

"""
    BinaryBases(states, particle::ℕ)
    BinaryBases(nstate::Integer, particle::ℕ)

Construct a set of binary bases that preserves the particle number conservation.
"""
@inline BinaryBases(nstate::Integer, particle::ℕ) = BinaryBases(one(nstate):nstate, particle)
function BinaryBases(states, particle::ℕ)
    stategroup = BinaryBasis(states)
    table = BinaryBasis{basistype(eltype(states))}[]
    table!(table, NTuple{length(states), basistype(eltype(states))}(sort!(collect(states); rev=true)), Val(value(particle)))
    return BinaryBases([particle], [stategroup], table)
end
function table!(table, states::Tuple, ::Val{N}) where N
    for poses in Combinations{N}(states)
        push!(table, BinaryBasis{eltype(states)}(poses))
    end
    return reverse!(table)
end

"""
    BinaryBases(spindws, spinups, sz::𝕊ᶻ)

Construct a set of binary bases that preserves the spin z-component but not the particle number conservation.
"""
function BinaryBases(spindws, spinups, sz::𝕊ᶻ)
    stategroup = BinaryBasis([spindws..., spinups...])
    basistable = typeof(stategroup)[]
    for nup in max(Int(2*value(sz)), 0):min(length(spinups)+Int(2*value(sz)), length(spinups))
        ndw = nup-Int(2*value(sz))
        append!(basistable, BinaryBases(spindws, ℕ(ndw)) ⊗ BinaryBases(spinups, ℕ(nup)))
    end
    return BinaryBases([sz], [stategroup], sort!(basistable)::Vector{typeof(stategroup)})
end

"""
    BinaryBases(spindws, spinups, spinfulparticle::Abelian[ℕ ⊠ 𝕊ᶻ])
    BinaryBases(spindws, spinups, spinfulparticle::Abelian[𝕊ᶻ ⊠ ℕ])

Construct a set of binary bases that preserves both the particle number and the spin z-component conservation.
"""
function BinaryBases(spindws, spinups, spinfulparticle::Abelian[ℕ ⊠ 𝕊ᶻ])
    ndw = Int(values(spinfulparticle)[1]/2-values(spinfulparticle)[2])
    nup = Int(values(spinfulparticle)[1]/2+values(spinfulparticle)[2])
    basesdw = BinaryBases(spindws, ℕ(ndw)) ⊠ 𝕊ᶻ(-ndw//2)
    basesup = BinaryBases(spinups, ℕ(nup)) ⊠ 𝕊ᶻ(nup//2)
    return basesdw ⊗ basesup
end
function BinaryBases(spindws, spinups, spinfulparticle::Abelian[𝕊ᶻ ⊠ ℕ])
    ndw = Int(values(spinfulparticle)[2]/2-values(spinfulparticle)[1])
    nup = Int(values(spinfulparticle)[2]/2+values(spinfulparticle)[1])
    basesdw = 𝕊ᶻ(-ndw//2) ⊠ BinaryBases(spindws, ℕ(ndw))
    basesup = 𝕊ᶻ(nup//2) ⊠ BinaryBases(spinups, ℕ(nup))
    return basesdw ⊗ basesup
end

"""
    ⊠(bs::BinaryBases, another::Abelian) -> BinaryBases
    ⊠(another::Abelian, bs::BinaryBases) -> BinaryBases

Deligne tensor product the quantum number of a set of binary bases with another quantum number.
"""
@inline ⊠(bs::BinaryBases, another::Abelian) = BinaryBases([qn ⊠ another for qn in bs.quantumnumbers], bs.stategroups, bs.table)
@inline ⊠(another::Abelian, bs::BinaryBases) = BinaryBases([another ⊠ qn for qn in bs.quantumnumbers], bs.stategroups, bs.table)

"""
    ⊗(bs₁::BinaryBases, bs₂::BinaryBases) -> BinaryBases

Get the direct product of two sets of binary bases.
"""
function ⊗(bs₁::BinaryBases, bs₂::BinaryBases)
    @assert productable(bs₁, bs₂) "⊗ error: the input two sets of bases cannot be direct producted."
    table = Vector{promote_type(eltype(bs₁), eltype(bs₂))}(undef, length(bs₁)*length(bs₂))
    count = 1
    for (b₁, b₂) in product(bs₁, bs₂)
        table[count] = b₁ ⊗ b₂
        count += 1
    end
    quantumnumbers = [bs₁.quantumnumbers; bs₂.quantumnumbers]
    stategroups = [bs₁.stategroups; bs₂.stategroups]
    permutation = sortperm(stategroups)
    return BinaryBases(permute!(quantumnumbers, permutation), permute!(stategroups, permutation), sort!(table))
end

"""
    matrix(op::Operator, braket::NTuple{2, BinaryBases}, table::AbstractDict, dtype=valtype(op); kwargs...) -> SparseMatrixCSC{dtype, Int}

Get the CSC-formed sparse matrix representation of an operator.

Here, `table` specifies the order of the operator indexes.
"""
function matrix(op::Operator, braket::NTuple{2, BinaryBases}, table::AbstractDict, dtype=valtype(op); kwargs...)
    bra, ket = braket
    @assert match(bra, ket) "matrix error: mismatched bra and ket."
    ndata, intermediate = 1, zeros(ket|>eltype, length(op)+1)
    data, indices, indptr = zeros(dtype, dimension(ket)), zeros(Int, dimension(ket)), zeros(Int, dimension(ket)+1)
    sequences = NTuple{length(op), Int}(table[op[i]] for i in reverse(1:length(op)))
    iscreations = NTuple{length(op), Bool}(iscreation(index) for index in reverse(id(op)))
    for i = 1:dimension(ket)
        flag = true
        indptr[i] = ndata
        intermediate[1] = ket[i]
        for j = 1:length(op)
            isone(intermediate[j], sequences[j])==iscreations[j] && (flag = false; break)
            intermediate[j+1] = iscreations[j] ? one(intermediate[j], sequences[j]) : zero(intermediate[j], sequences[j])
        end
        if flag
            nsign = 0
            statistics(eltype(op))==:f && for j = 1:length(op)
                nsign += count(intermediate[j], 1, sequences[j]-1)
            end
            index = searchsortedfirst(intermediate[end], bra)
            if index<=dimension(bra) && bra[index]==intermediate[end]
                indices[ndata] = index
                data[ndata] = op.value*(-1)^nsign
                ndata += 1
            end
        end
    end
    indptr[end] = ndata
    return SparseMatrixCSC(dimension(bra), dimension(ket), indptr, indices[1:ndata-1], data[1:ndata-1])
end

## ED based on binary bases for canonical fermionic and hardcore bosonic systems
"""
    EDKind(::Type{<:FockIndex})

Kind of the exact diagonalization method applied to a canonical quantum Fock lattice system.
"""
@inline EDKind(::Type{<:FockIndex}) = EDKind(:Binary)

"""
    Metric(::EDKind{:Binary}, ::Hilbert{<:Fock}) -> OperatorIndexToTuple

Get the index-to-tuple metric for a canonical quantum Fock lattice system.
"""
@inline @generated Metric(::EDKind{:Binary}, ::Hilbert{<:Fock}) = OperatorIndexToTuple(:spin, :site, :orbital)

"""
    Sector(hilbert::Hilbert{<:Fock}, basistype::Type{<:Unsigned}=UInt) -> BinaryBases
    Sector(quantumnumber::ℕ, hilbert::Hilbert{<:Fock}, basistype::Type{<:Unsigned}=UInt; table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))) -> BinaryBases
    Sector(quantumnumber::Union{𝕊ᶻ, Abelian[ℕ ⊠ 𝕊ᶻ], Abelian[𝕊ᶻ ⊠ ℕ]}, hilbert::Hilbert{<:Fock}, basistype::Type{<:Unsigned}=UInt; table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))) -> BinaryBases

Construct the binary bases of a Hilbert space with the specified quantum number.
"""
@inline Sector(hilbert::Hilbert{<:Fock}, basistype::Type{<:Unsigned}=UInt) = BinaryBases(basistype(sum([length(internal)÷2 for internal in values(hilbert)])))
@inline function Sector(quantumnumber::ℕ, hilbert::Hilbert{<:Fock}, basistype::Type{<:Unsigned}=UInt; table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert)))
    states = Set{basistype}(table[Index(site, index)] for (site, internal) in hilbert for index in internal)
    return BinaryBases(states, quantumnumber)
end
@inline function Sector(quantumnumber::Union{𝕊ᶻ, Abelian[ℕ ⊠ 𝕊ᶻ], Abelian[𝕊ᶻ ⊠ ℕ]}, hilbert::Hilbert{<:Fock}, basistype::Type{<:Unsigned}=UInt; table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert)))
    @assert all(internal->internal.nspin==2, values(hilbert)) "Sector error: only for spin-1/2 systems."
    spindws = Set{basistype}(table[Index(site, index)] for (site, internal) in hilbert for index in internal if index.spin==-1//2)
    spinups = Set{basistype}(table[Index(site, index)] for (site, internal) in hilbert for index in internal if index.spin==+1//2)
    return BinaryBases(spindws, spinups, quantumnumber)
end

# Abelian bases, used for canonical spin systems by default
## Basics for Abelian bases
"""
    partition(n::Int) -> NTuple{2, Vector{Int}}

Get the default partition of n local Hilbert spaces.
"""
@inline partition(n::Int) =(cut=n÷2; (collect(1:cut), collect(cut+1:n)))

"""
    AbelianBases{A<:Abelian, N} <: Sector

A set of Abelian bases, that is, a set of bases composed from the product of local Abelian Graded spaces.

To improve the efficiency of the product of local Abelian Graded spaces, we adopt a two-step strategy:
1) partition the local spaces into several groups in each of which the local spaces are direct producted and rearranged according to the Abelian quantum numbers, and then 
2) glue the results obtained in the previous step so that a sector with a certain Abelian quantum number can be targeted.

In principle, a binary-tree strategy can be more efficient, but our two-step strategy is enough for a quantum system that can be solved by the exact diagonalization method.

The partition of the local Abelian Graded spaces is assigned by a `NTuple{N, Vector{Int}}`, with each of its element contains the sequences of the grouped local spaces specified by a table.
"""
struct AbelianBases{A<:Abelian, N} <: Sector
    quantumnumber::A
    locals::Vector{Graded{A}}
    partition::NTuple{N, Vector{Int}}
    gradeds::NTuple{N, Graded{A}}
    permutations::NTuple{N, Vector{Int}}
    record::Dict{NTuple{N, A}, Int}
    dim::Int
end
@inline id(bs::AbelianBases) = (bs.quantumnumber, bs.locals, bs.partition)
@inline dimension(bs::AbelianBases) = bs.dim
function Base.show(io::IO, bs::AbelianBases)
    @printf io "%s" "{"
    for (i, positions) in enumerate(bs.partition)
        @printf io "[%s]" join([tostr(bs.locals[position], position) for position in positions], "⊗")
        i<length(bs.partition) && @printf io "%s" " ⊗ "
    end
    @printf io ": %s}" bs.quantumnumber
end
@inline tostr(internal::Graded, order::Int) = string(internal, join('₀'+d for d in reverse(digits(order))))
function Base.match(bs₁::BS, bs₂::BS) where {BS<:AbelianBases}
    bs₁.locals==bs₂.locals || return false
    for (positions₁, positions₂) in zip(bs₁.partition, bs₂.partition)
        positions₁==positions₂ || return false
    end
    return true
end
@inline sumable(bs₁::BS, bs₂::BS) where {BS<:AbelianBases} = Abelian(bs₁) ≠ Abelian(bs₂)

"""
    Abelian(bs::AbelianBases)

Get the Abelian quantum number of a set of spin bases.
"""
@inline Abelian(bs::AbelianBases) = bs.quantumnumber

"""
    range(bs::AbelianBases) -> AbstractVector{Int}

Get the range of the target sector of an `AbelianBases` in the direct producted bases.
"""
@inline Base.range(bs::AbelianBases{ℤ₁}) = 1:dimension(bs)
function Base.range(bs::AbelianBases)
    result = Int[]
    dims = reverse(map(dimension, bs.gradeds))
    cartesian, linear = CartesianIndices(dims), LinearIndices(dims)
    total, slice = decompose(⊗(bs.gradeds...))
    for i in slice[range(total, bs.quantumnumber)]
        push!(result, linear[map(getindex, reverse(bs.permutations), cartesian[i].I)...])
    end
    return result
end

"""
    AbelianBases(locals::AbstractVector{Int}, partition::NTuple{N, AbstractVector{Int}}=partition(length(locals))) where N

Construct a set of spin bases that subjects to no quantum number conservation.
"""
@inline function AbelianBases(locals::AbstractVector{Int}, partition::NTuple{N, AbstractVector{Int}}=partition(length(locals))) where N
    return AbelianBases(Graded{ℤ₁}[Graded{ℤ₁}(0=>dim) for dim in locals], ℤ₁(0), partition)
end

"""
    AbelianBases(locals::Vector{Graded{A}}, quantumnumber::A, partition::NTuple{N, AbstractVector{Int}}=partition(length(locals))) where {N, A<:Abelian}

Construct a set of spin bases that preserves a certain symmetry specified by the corresponding Abelian quantum number.
"""
function AbelianBases(locals::Vector{Graded{A}}, quantumnumber::A, partition::NTuple{N, AbstractVector{Int}}=partition(length(locals))) where {N, A<:Abelian}
    gradeds, permutations, total, records = intermediate(locals, partition, quantumnumber)
    return AbelianBases{A, N}(quantumnumber, locals, partition, gradeds, permutations, records[1], dimension(total, quantumnumber))
end
function intermediate(spins::Vector{Graded{A}}, partition::NTuple{N, AbstractVector{Int}}, quantumnumbers::A...) where {A<:Abelian, N}
    gradeds, permutations = Graded{A}[], Vector{Int}[]
    for positions in partition
        if length(positions)>0
            graded, permutation = decompose(⊗([spins[position] for position in positions]...))
            push!(gradeds, graded)
            push!(permutations, permutation)
        end
    end
    graded = ⊗(NTuple{N, Graded{A}}(gradeds)...)
    total, fusion = merge(graded)
    records = map(quantumnumbers) do quantumnumber
        count = 1
        record = Dict{NTuple{N, A}, Int}()
        for qns in fusion[quantumnumber]
            record[qns] = count
            count += dimension(graded, qns)
        end
        record
    end
    return NTuple{N, Graded{A}}(gradeds), NTuple{N, Vector{Int}}(permutations), total::Graded{A}, records
end

"""
    matrix(index::OperatorIndex, graded::Graded, dtype::Type{<:Number}=ComplexF64) -> Matrix{dtype}

Get the matrix representation of an `OperatorIndex` on an Abelian graded space.
"""
@inline matrix(index::OperatorIndex, graded::Graded, dtype::Type{<:Number}=ComplexF64) = matrix(InternalIndex(index), graded, dtype)

"""
    matrix(op::Operator, braket::NTuple{2, AbelianBases}, table::AbstractDict, dtype=valtype(op); kwargs...) -> SparseMatrixCSC{dtype, Int}

Get the CSC-formed sparse matrix representation of an operator.

Here, `table` specifies the order of the operator indexes.
"""
function matrix(op::Operator, braket::NTuple{2, AbelianBases{ℤ₁}}, table::AbstractDict, dtype=valtype(op); kwargs...)
    bra, ket = braket
    @assert match(bra, ket) "matrix error: mismatched bra and ket."
    ms = matrices(op, ket.locals, table, dtype)
    intermediate = eltype(ms)[]
    for positions in ket.partition
        for (i, position) in enumerate(positions)
            if i==1
                push!(intermediate, ms[position])
            else
                intermediate[end] = kron(intermediate[end], ms[position])
            end
        end
    end
    result = intermediate[1]
    for i = 2:length(intermediate)
        result = kron(result, intermediate[i])
    end
    return result
end
function matrix(op::Operator, braket::NTuple{2, AbelianBases}, table::AbstractDict, dtype=valtype(op); kwargs...)
    bra, ket = braket
    @assert match(bra, ket) "matrix error: mismatched bra and ket."
    ms = matrices(op, ket.locals, table, dtype)
    intermediate = map((positions, graded, permutation)->blocks(ms[positions], graded, permutation), ket.partition, ket.gradeds, ket.permutations)
    result = SparseMatrixCOO(Int[], Int[], dtype[], dimension(bra), dimension(ket))
    for (row_keys, row_start) in pairs(bra.record)
        for (col_keys, col_start) in pairs(ket.record)
            kron!(result, map((row_key, col_key, m)->m[(row_key, col_key)], row_keys, col_keys, intermediate); origin=(row_start, col_start))
        end
    end
    return SparseMatrixCSC(result)
end
function matrices(op::Operator, locals::Vector{<:Graded}, table::AbstractDict, dtype)
    result = [sparse(one(dtype)*I, dimension(internal), dimension(internal)) for internal in locals]
    for (i, index) in enumerate(op)
        position = table[index]
        i==1 && (result[position] *= op.value)
        result[position] *= sparse(matrix(index, locals[position], dtype))
    end
    return result
end
function blocks(ms::Vector{<:SparseMatrixCSC}, graded::Graded, permutation::Vector{Int})
    result = Dict{Tuple{eltype(graded), eltype(graded)}, SparseMatrixCOO{eltype(eltype(ms)), Int}}()
    m = permute!(reduce(kron, ms), permutation, permutation)
    rows, vals = rowvals(m), nonzeros(m)
    for j = 1:length(graded)
        temp = [(Int[], Int[], eltype(eltype(ms))[]) for i=1:length(graded)]
        for (k, col) in enumerate(range(graded, j))
            pos = 1
            for index in nzrange(m, col)
                row = rows[index]
                val = vals[index]
                pos = findindex(row, graded, pos)
                push!(temp[pos][1], row-cumsum(graded, pos-1))
                push!(temp[pos][2], k)
                push!(temp[pos][3], val)
            end
        end
        for (i, (is, js, vs)) in enumerate(temp)
            result[(graded[i], graded[j])] = SparseMatrixCOO(is, js, vs, dimension(graded, i), dimension(graded, j))
        end
    end
    return result
end
function Base.kron!(result::SparseMatrixCOO, ms::Tuple{Vararg{SparseMatrixCOO}}; origin::Tuple{Int, Int}=(1, 1))
    ms = reverse(ms)
    row_linear = LinearIndices(map(m->1:size(m)[1], ms))
    col_linear = LinearIndices(map(m->1:size(m)[2], ms))
    for indexes in product(map(m->1:nnz(m), ms)...)
        push!(result.is, row_linear[map((index, m)->m.is[index], indexes, ms)...] + origin[1] - 1)
        push!(result.js, col_linear[map((index, m)->m.js[index], indexes, ms)...] + origin[2] - 1)
        push!(result.vs, prod(map((index, m)->m.vs[index], indexes, ms)))
    end
    return result
end

## ED based on Abelian bases for canonical spin systems
"""
    Graded{ℤ₁}(spin::Spin)
    Graded{𝕊ᶻ}(spin::Spin)

Decompose a local spin space into an Abelian graded space that preserves 1) no symmetry, and 2) spin-z component symmetry.
"""
@inline Graded{ℤ₁}(spin::Spin) = Graded{ℤ₁}(0=>Int(2*totalspin(spin)+1))
@inline Graded{𝕊ᶻ}(spin::Spin) = Graded{𝕊ᶻ}(sz=>1 for sz in -totalspin(spin):1:totalspin(spin))'

"""
    matrix(index::SpinIndex, graded::Graded, dtype::Type{<:Number}=ComplexF64) -> Matrix{dtype}

Get the matrix representation of a `SpinIndex` on an Abelian graded space.
"""
@inline function matrix(index::SpinIndex, graded::Graded{ℤ₁}, dtype::Type{<:Number}=ComplexF64)
    @assert Int(2*totalspin(index)+1)==dimension(graded) "matrix error: mismatched spin index and Abelian graded space."
    return matrix(index, dtype)
end
@inline function matrix(index::SpinIndex, graded::Graded{𝕊ᶻ}, dtype::Type{<:Number}=ComplexF64)
    S = totalspin(index)
    @assert Int(2S+1)==dimension(graded)==length(graded) && first(graded)==𝕊ᶻ(S) && last(graded)==𝕊ᶻ(-S) "matrix error: mismatched spin index and Abelian graded space."
    return matrix(index, dtype)
end

"""
    EDKind(::Type{<:SpinIndex})

Kind of the exact diagonalization method applied to a canonical quantum spin lattice system.
"""
@inline EDKind(::Type{<:SpinIndex}) = EDKind(:Abelian)

"""
    Metric(::EDKind{:Abelian}, ::Hilbert{<:Spin}) -> OperatorIndexToTuple

Get the index-to-tuple metric for a canonical quantum spin lattice system.
"""
@inline @generated Metric(::EDKind{:Abelian}, ::Hilbert{<:Spin}) = OperatorIndexToTuple(:site)

"""
    Sector(hilbert::Hilbert{<:Spin}, partition::Tuple{Vararg{AbstractVector{Int}}}=partition(length(hilbert))) -> AbelianBases
    Sector(
        quantumnumber::Abelian, hilbert::Hilbert{<:Spin}, partition::Tuple{Vararg{AbstractVector{Int}}}=partition(length(hilbert));
        table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))
    ) -> AbelianBases

Construct the Abelian bases of a spin Hilbert space with the specified quantum number.
"""
@inline function Sector(hilbert::Hilbert{<:Spin}, partition::Tuple{Vararg{AbstractVector{Int}}}=partition(length(hilbert)))
    return Sector(ℤ₁(0), hilbert, partition; table=Table(hilbert, Metric(EDKind(hilbert), hilbert)))
end
@inline function Sector(
    quantumnumber::Abelian, hilbert::Hilbert{<:Spin}, partition::Tuple{Vararg{AbstractVector{Int}}}=partition(length(hilbert));
    table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))
)
    @assert sort(vcat(partition...))==1:length(hilbert) "Sector error: incorrect partition."
    return AbelianBases(sorted_locals(typeof(quantumnumber), hilbert, table), quantumnumber, partition)
end
@inline function sorted_locals(::Type{A}, hilbert::Hilbert, table::AbstractDict) where {A<:Abelian}
    result = Graded{A}[Graded{A}(internal) for internal in values(hilbert)]
    sites = collect(keys(hilbert))
    perm = sortperm(sites; by=site->table[𝕊(site, :α)])
    return permute!(result, perm)
end

"""
    broadcast(
        ::Type{Sector}, quantumnumbers::OneAtLeast{Abelian}, hilbert::Hilbert{<:Spin}, partition::NTuple{N, AbstractVector{Int}}=partition(length(hilbert));
        table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))
    ) where N -> NTuple{fieldcount(typeof(quantumnumbers)), AbelianBases}

Construct a set of Abelian based based on the quantum numbers and a Hilbert space.
"""
function Base.broadcast(
    ::Type{Sector}, quantumnumbers::OneAtLeast{A}, hilbert::Hilbert{<:Spin}, partition::NTuple{N, AbstractVector{Int}}=partition(length(hilbert));
    table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))
) where {N, A<:Abelian}
    @assert sort(vcat(partition...))==1:length(hilbert) "Broadcast of Sector error: incorrect partition."
    quantumnumbers = OneOrMore(quantumnumbers)
    locals = sorted_locals(A, hilbert, table)
    gradeds, permutations, total, records = intermediate(locals, partition, quantumnumbers...)
    return map(quantumnumbers, records) do quantumnumber, record
        AbelianBases{A, N}(quantumnumber, locals, partition, gradeds, permutations, record, dimension(total, quantumnumber))
    end
end

"""
    StaticSpinStructureFactor{R<:ReciprocalSpace} <: Action

Static spin structure factor.
"""
struct StaticSpinStructureFactor{R<:ReciprocalSpace} <: Action
    reciprocalspace::R
end
@inline options(::Type{<:Assignment{<:StaticSpinStructureFactor}}) = basicoptions

"""
    StaticSpinStructureFactorData{R<:ReciprocalSpace, V<:Array{Float64}} <: Data

Data of static spin structure factor, including:

1) `reciprocalspace::R`: reciprocal space to compute the static spin structure factor.
2) `values::V`: values of static spin structure factor.
"""
struct StaticSpinStructureFactorData{R<:ReciprocalSpace, V<:Array{Float64}} <: Data
    reciprocalspace::R
    values::V
end
function run!(ed::Algorithm{<:ED}, factor::Assignment{<:StaticSpinStructureFactor}; options...)
    eigensystem = only(factor.dependencies)
    @assert isa(eigensystem, Assignment{<:EDEigen}) "run! error: wrong dependencies."
    lattice = ed.frontend.lattice
    hilbert = ed.frontend.system.hilbert
    table = ed.frontend.matrixization.table
    Ω = only(eigensystem.data.vectors)
    sector = only(eigensystem.data.sectors)
    len = length(factor.action.reciprocalspace)
    result = initialization(factor.action.reciprocalspace)
    for i = 1:length(lattice), j = 1:length(lattice)
        x = dot(Ω, matrix(Operator(1, Index(i, hilbert[i][1]), Index(j, hilbert[j][1])), (sector, sector), table, ComplexF64), Ω)
        y = dot(Ω, matrix(Operator(1, Index(i, hilbert[i][2]), Index(j, hilbert[j][2])), (sector, sector), table, ComplexF64), Ω)
        z = dot(Ω, matrix(Operator(1, Index(i, hilbert[i][3]), Index(j, hilbert[j][3])), (sector, sector), table, ComplexF64), Ω)
        displacement = lattice[i] - lattice[j]
        for (k, momentum) in enumerate(factor.action.reciprocalspace)
            phase = exp(1im*dot(momentum, displacement))
            result[k] += real(x*phase)
            result[k+len] += real(y*phase)
            result[k+2len] += real(z*phase)
        end
    end
    for k in eachindex(result)
        result[k] /= len
    end
    return  StaticSpinStructureFactorData(factor.action.reciprocalspace, result)
end
@inline initialization(reciprocalspace::Union{BrillouinZone, ReciprocalZone}) = zeros(Float64, map(length, reverse(shape(reciprocalspace)))..., 3)
@inline initialization(reciprocalspace::ReciprocalSpace) = zeros(Float64, length(reciprocalspace), 3)
