module EDCore

using Arpack: eigs
using LinearAlgebra: Eigen
using QuantumLattices: expand, id, idtype, reparameter
using QuantumLattices: AbstractLattice, Frontend, Image, LinearTransformation, MatrixRepresentation, Operator, OperatorGenerator, OperatorPack, Operators, OperatorSum, OperatorUnit, Term, VectorSpace, VectorSpaceEnumerative, VectorSpaceStyle
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
@inline EDKind(::Type{T}) where {T<:Term} = error("EDKind error: not defined for $(kind(T)).")
@inline @generated function EDKind(::Type{TS}) where {TS<:Tuple{Vararg{Term}}}
    exprs = []
    for i = 1:fieldcount(TS)
        push!(exprs, :(typeof(EDKind(fieldtype(TS, $i)))))
    end
    return Expr(:call, Expr(:call, :reduce, :promote_type, Expr(:tuple, exprs...)))
end

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
    matrix(ed::ED, sector::Sector; kwargs...) -> EDMatrix
    matrix(ed::ED, sector=first(ed.Hₘ.transformation.brakets); kwargs...) -> EDMatrix

Get the sparse matrix representation of a quantum lattice system in a sector of the target space.
"""
@inline matrix(ed::ED, sector::Sector; kwargs...) = matrix(ed, (sector, sector); kwargs...)
@inline function matrix(ed::ED, braket::NTuple{2, Sector}=first(ed.Hₘ.transformation.brakets); kwargs...)
    return expand(SectorFilter(braket)(ed.Hₘ))[braket]
end

"""
    eigen(m::EDMatrix; nev=6, which=:SR, tol=0.0, maxiter=300, sigma=nothing, v₀=dtype(m)[]) -> Eigen

Solve the eigen problem by the restarted Lanczos method provided by the Arpack package.
"""
@inline function eigen(m::EDMatrix; nev=6, which=:SR, tol=0.0, maxiter=300, sigma=nothing, v₀=dtype(m)[])
    if size(m.matrix)[1] > 1
        eigvals, eigvecs = eigs(m.matrix; nev=nev, which=which, tol=tol, maxiter=maxiter, sigma=sigma, ritzvec=true, v0=v₀)
    else
        eigvals, eigvecs = eigen(collect(m.matrix))
    end
    return Eigen(eigvals, eigvecs)
end

end # module
