module EDCore

using Arpack: eigs
using LinearAlgebra: Eigen, Factorization
using QuantumLattices: plain, bonds, expand, id, idtype, reparameter
using QuantumLattices: AbstractLattice, Boundary, Frontend, Hilbert, Image, LinearTransformation, MatrixRepresentation, Metric, Neighbors, Operator, OperatorGenerator, OperatorPack, Operators, OperatorSum, OperatorUnit, Table, Term, VectorSpace, VectorSpaceEnumerative, VectorSpaceStyle
using SparseArrays: SparseMatrixCSC

import LinearAlgebra: eigen
import QuantumLattices: Parameters, ⊕, add!, contentnames, dtype, getcontent, kind, matrix, parameternames, statistics, update!

export ED, EDKind, EDMatrix, EDMatrixRepresentation, Sector, SectorFilter, TargetSpace

"""
    abstract type Sector <: OperatorUnit

A sector of the Hilbert space which forms the bases of an irreducible representation of the Hamiltonian of a quantum lattice system.
"""
abstract type Sector <: OperatorUnit end

"""
    TargetSpace{S<:Sector} <: VectorSpace{S}

The target Hilbert space in which the exact diagonalization method is performed, which could be the direct sum of several sectors.
"""
struct TargetSpace{S<:Sector} <: VectorSpace{S}
    sectors::Vector{S}
end
@inline VectorSpaceStyle(::Type{<:TargetSpace}) = VectorSpaceEnumerative()
@inline contentnames(::Type{<:TargetSpace}) = (:contents,)
@inline getcontent(target::TargetSpace, ::Val{:contents}) = target.sectors
function add!(target::TargetSpace, sector::Sector)
    push!(target.sectors, sector)
    return target
end
function add!(target::TargetSpace, another::TargetSpace)
    for sector in another
        add!(target, sector)
    end
    return target
end

"""
    ⊕(sector::Sector, sectors::Union{Sector, TargetSpace}...) -> TargetSpace
    ⊕(target::TargetSpace, sectors::Union{Sector, TargetSpace}...) -> TargetSpace

Get the direct sum of sectors and target spaces.
"""
@inline function ⊕(sector::Sector, sectors::Union{Sector, TargetSpace}...)
    result = TargetSpace(sector)
    map(op->add!(result, op), sectors)
    return result
end
@inline function ⊕(target::TargetSpace, sectors::Union{Sector, TargetSpace}...)
    result = TargetSpace(copy(target.sectors))
    map(op->add!(result, op), sectors)
    return result
end

"""
    TargetSpace(sector::Sector, sectors::Sector...)

Construct a target space from sectors.
"""
@inline function TargetSpace(sector::Sector, sectors::Sector...)
    result = TargetSpace([sector])
    for sec in sectors
        add!(result, sec)
    end
    return result
end

"""
    TargetSpace(hilbert::Hilbert; kwargs...)
    TargetSpace(hilbert::Hilbert, quantumnumbers::Tuple; table=Table(hilbert, Metric(EDKind(hilbert), hilbert)), kwargs...)
    TargetSpace(hilbert::Hilbert, quantumnumbers...; table=Table(hilbert, Metric(EDKind(hilbert), hilbert)), kwargs...)

Construct a target space from the total Hilbert space and the associated quantum numbers.
"""
@inline TargetSpace(hilbert::Hilbert; kwargs...) = TargetSpace(Sector(hilbert; kwargs...))
@inline TargetSpace(hilbert::Hilbert, quantumnumbers::Tuple; table=Table(hilbert, Metric(EDKind(hilbert), hilbert)), kwargs...) = TargetSpace(hilbert, quantumnumbers...; table=table, kwargs...)
@inline TargetSpace(hilbert::Hilbert, quantumnumbers...; table=Table(hilbert, Metric(EDKind(hilbert), hilbert)), kwargs...) = TargetSpace(map(quantumnumber->Sector(hilbert, quantumnumber; table=table, kwargs...), quantumnumbers)...)

# Generic exact diagonalization method
"""
    EDMatrix{S<:Sector, M<:SparseMatrixCSC} <: OperatorPack{M, Tuple{S, S}}

Matrix representation of quantum operators between a ket and a bra Hilbert space.
"""
struct EDMatrix{S<:Sector, M<:SparseMatrixCSC} <: OperatorPack{M, Tuple{S, S}}
    bra::S
    ket::S
    matrix::M
end
@inline parameternames(::Type{<:EDMatrix}) = (:sector, :value)
@inline getcontent(m::EDMatrix, ::Val{:id}) = (m.bra, m.ket)
@inline getcontent(m::EDMatrix, ::Val{:value}) = m.matrix
@inline dtype(::Type{<:EDMatrix{<:Sector, M}}) where {M<:SparseMatrixCSC} = eltype(M)
@inline EDMatrix(m::SparseMatrixCSC, braket::NTuple{2, Sector}) = EDMatrix(braket[1], braket[2], m)
@inline Base.promote_rule(M::Type{<:EDMatrix}, N::Type{<:Number}) = reparameter(M, :value, reparameter(valtype(M), 1, promote_type(dtype(M), N)))

"""
    EDMatrix(sector::Sector, m::SparseMatrixCSC)

Construct a matrix representation when the ket and bra spaces share the same bases.
"""
@inline EDMatrix(sector::Sector, m::SparseMatrixCSC) = EDMatrix(sector, sector, m)

"""
    eigen(m::EDMatrix; nev=6, which=:SR, tol=0.0, maxiter=300, sigma=nothing, v₀=dtype(m)[]) -> Eigen

Solve the eigen problem by the restarted Lanczos method provided by the Arpack package.
"""
@inline function eigen(m::EDMatrix; nev=6, which=:SR, tol=0.0, maxiter=300, sigma=nothing, v₀=dtype(m)[])
    @assert m.bra==m.ket "eigen error: eigen decomposition of an `EDMatrix` are only available for those with the same bra and ket spaces."
    if size(m.matrix)[1] > 1
        eigvals, eigvecs = eigs(m.matrix; nev=nev, which=which, tol=tol, maxiter=maxiter, sigma=sigma, ritzvec=true, v0=v₀)
    else
        eigvals, eigvecs = eigen(collect(m.matrix))
    end
    return Eigen(eigvals, eigvecs)
end

"""
    EDEigen{V<:Number, T<:Number, S<:Sector} <: Factorization{T}

Eigen decomposition in exact diagonalization method.

Compared to the usual eigen decomposition `Eigen`, `EDEigen` contains a `:sectors` attribute to store the sectors of Hilbert space in which the eigen values and eigen vectors are computed.
Furthermore, given that in different sectors the dimensions of the sub-Hilbert spaces can also be different, the `:vectors` attribute of `EDEigen` is a vector of vector instead of a matrix.
"""
struct EDEigen{V<:Number, T<:Number, S<:Sector} <: Factorization{T}
    values::Vector{V}
    vectors::Vector{Vector{T}}
    sectors::Vector{S}
end
@inline Base.iterate(content::EDEigen) = (content.values, Val(:vectors))
@inline Base.iterate(content::EDEigen, ::Val{:vectors}) = (content.vectors, Val(:sectors))
@inline Base.iterate(content::EDEigen, ::Val{:sectors}) = (content.sectors, Val(:done))
@inline Base.iterate(content::EDEigen, ::Val{:done}) = nothing

"""
    eigen(ms::OperatorSum{<:EDMatrix}; nev::Int=6, tol::Real=0.0, maxiter::Int=300, v₀::Union{AbstractVector, Dict{<:Sector, <:AbstractVector}}=dtype(eltype(ms))[]) -> EDEigen

Solve the eigen problem by the restarted Lanczos method provided by the Arpack package.
"""
@inline function eigen(ms::OperatorSum{<:EDMatrix}; nev::Int=6, tol::Real=0.0, maxiter::Int=300, v₀::Union{AbstractVector, Dict{<:Sector, <:AbstractVector}}=dtype(eltype(ms))[])
    isa(v₀, AbstractVector) && (v₀ = Dict(m.ket=>v₀ for m in ms))
    values, vectors, sectors = real(dtype(eltype(ms)))[], Vector{dtype(eltype(ms))}[], eltype(idtype(eltype(ms)))[]
    for m in ms
        k = min(length(m.ket), nev)
        eigensystem = eigen(m; nev=k, which=:SR, tol=tol, maxiter=maxiter, v₀=v₀[m.ket])
        for i = 1:k
            push!(values, eigensystem.values[i])
            push!(vectors, eigensystem.vectors[:, i])
            push!(sectors, m.ket)
        end
    end
    perm = sortperm(values)
    return EDEigen(values[perm], vectors[perm], sectors[perm])
end

"""
    EDMatrixRepresentation{S<:Sector, T} <: MatrixRepresentation

Exact matrix representation of a quantum lattice system on a target Hilbert space.
"""
struct EDMatrixRepresentation{S<:Sector, T} <: MatrixRepresentation
    brakets::Vector{NTuple{2, S}}
    table::T
end
@inline function Base.valtype(::Type{<:EDMatrixRepresentation{S}}, M::Type{<:Operator}) where {S<:Sector}
    M = EDMatrix{S, SparseMatrixCSC{valtype(M), Int}}
    return OperatorSum{M, idtype(M)}
end
@inline Base.valtype(R::Type{<:EDMatrixRepresentation}, M::Type{<:Operators}) = valtype(R, eltype(M))
function (representation::EDMatrixRepresentation)(m::Operator; kwargs...)
    result = zero(valtype(representation, m))
    for braket in representation.brakets
        add!(result, EDMatrix(matrix(m, braket, representation.table; kwargs...), braket))
    end
    return result
end

"""
    EDMatrixRepresentation(target::TargetSpace, table)

Construct a exact matrix representation.
"""
@inline EDMatrixRepresentation(target::TargetSpace, table) = EDMatrixRepresentation([(sector, sector) for sector in target], table)

"""
    SectorFilter{S} <: LinearTransformation

Filter the target bra and ket Hilbert spaces.
"""
struct SectorFilter{S} <: LinearTransformation
    brakets::Set{NTuple{2, S}}
end
@inline Base.valtype(::Type{<:SectorFilter}, M::Type{<:OperatorSum{<:EDMatrix}}) = M
@inline (sectorfileter::SectorFilter)(m::EDMatrix) = id(m)∈sectorfileter.brakets ? m : 0
@inline SectorFilter(sector::Sector, sectors::Sector...) = SectorFilter((sector, sector), map(op->(op, op), sectors)...)
@inline SectorFilter(braket::NTuple{2, Sector}, brakets::NTuple{2, Sector}...) = SectorFilter(push!(Set{typeof(braket)}(), braket, brakets...))

"""
    EDKind{K}

The kind of the exact diagonalization method applied to a quantum lattice system.
"""
struct EDKind{K} end
@inline EDKind(K::Symbol) = EDKind{K}()
@inline EDKind(hilbert::Hilbert) = EDKind(typeof(hilbert))
@inline EDKind(::Type{H}) where {H<:Hilbert} = error("EDKind error: not defined.")

"""
    ED{K<:EDKind, L<:AbstractLattice, G<:OperatorGenerator, M<:Image} <: Frontend

Exact diagonalization method of a quantum lattice system.
"""
struct ED{K<:EDKind, L<:AbstractLattice, G<:OperatorGenerator, M<:Image} <: Frontend
    lattice::L
    H::G
    Hₘ::M
    function ED{K}(lattice::AbstractLattice, H::OperatorGenerator, mr::EDMatrixRepresentation) where K
        Hₘ = mr(H)
        new{K, typeof(lattice), typeof(H), typeof(Hₘ)}(lattice, H, Hₘ)
    end
end
@inline kind(ed::ED) = kind(typeof(ed))
@inline kind(::Type{<:ED{K}}) where K = K()
@inline Base.valtype(::Type{<:ED{<:EDKind, <:AbstractLattice, G}}) where {G<:OperatorGenerator} = valtype(eltype(G))
@inline statistics(ed::ED) = statistics(typeof(ed))
@inline statistics(::Type{<:ED{<:EDKind, <:AbstractLattice, G}}) where {G<:OperatorGenerator} = statistics(eltype(eltype(G)))
@inline function update!(ed::ED; kwargs...)
    if length(kwargs)>0
        update!(ed.H; kwargs...)
        update!(ed.Hₘ, ed.H)
    end
    return ed
end
@inline Parameters(ed::ED) = Parameters(ed.H)

"""
    ED(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}, quantumnumbers::Tuple; neighbors::Union{Nothing, Int, Neighbors}=nothing, boundary::Boundary=plain, kwargs...)
    ED(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}, quantumnumbers...; neighbors::Union{Nothing, Int, Neighbors}=nothing, boundary::Boundary=plain, kwargs...)

Construct the exact diagonalization method for a canonical quantum Fock lattice system.
"""
function ED(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}, quantumnumbers::Tuple; neighbors::Union{Nothing, Int, Neighbors}=nothing, boundary::Boundary=plain, kwargs...)
    return ED(lattice, hilbert, terms, quantumnumbers...; neighbors=neighbors, boundary=boundary, kwargs...)
end
function ED(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}, quantumnumbers...; neighbors::Union{Nothing, Int, Neighbors}=nothing, boundary::Boundary=plain, kwargs...)
    k = EDKind(hilbert)
    table = Table(hilbert, Metric(k, hilbert))
    targetspace = TargetSpace(hilbert, quantumnumbers...; table=table, kwargs...)
    isnothing(neighbors) && (neighbors = maximum(term->term.bondkind, terms))
    H = OperatorGenerator(terms, bonds(lattice, neighbors), hilbert; half=false, boundary=boundary)
    mr = EDMatrixRepresentation(targetspace, table)
    return ED{typeof(k)}(lattice, H, mr)
end

"""
    matrix(ed::ED, sectors::Sector...; kwargs...) -> OperatorSum{<:EDMatrix}

Get the sparse matrix representation of a quantum lattice system in the target space.
"""
@inline matrix(ed::ED; kwargs...) = expand(ed.Hₘ)
@inline matrix(ed::ED, sectors::Sector...; kwargs...) = expand(SectorFilter(sectors...)(ed.Hₘ))

"""
    eigen(ed::ED, sectors::Sector...; kwargs...) -> EDEigen

Solve the eigen problem by the restarted Lanczos method provided by the Arpack package.
"""
@inline eigen(ed::ED, sectors::Sector...; kwargs...) = eigen(matrix(ed, sectors...); kwargs...)

end # module
