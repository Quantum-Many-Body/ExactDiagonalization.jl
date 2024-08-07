module EDCore

using Arpack: eigs
using LinearAlgebra: Eigen, Factorization, norm
using QuantumLattices: eager, plain, bonds, expand, id, idtype, reparameter
using QuantumLattices: AbelianNumber, AbstractLattice, Algorithm, Boundary, Entry, Frontend, Hilbert, Image, LinearTransformation, MatrixRepresentation, Metric, Neighbors, Operator, OperatorGenerator, OperatorPack, Operators, OperatorSum, OperatorUnit, Table, Term, VectorSpace, VectorSpaceEnumerative, VectorSpaceStyle, reset!
using SparseArrays: SparseMatrixCSC, spzeros
using TimerOutputs: TimerOutput, @timeit

import LinearAlgebra: eigen
import QuantumLattices: Parameters, ⊕, add!, contentnames, dtype, getcontent, kind, matrix, parameternames, prepare!, statistics, update!

export edtimer, ED, EDEigen, EDKind, EDMatrix, EDMatrixRepresentation, Sector, SectorFilter, TargetSpace, productable, release!, sumable

"""
    const edtimer = TimerOutput()

The default shared timer for all exact diagonalization methods.
"""
const edtimer = TimerOutput()

"""
    abstract type Sector <: OperatorUnit

A sector of the Hilbert space which forms the bases of an irreducible representation of the Hamiltonian of a quantum lattice system.
"""
abstract type Sector <: OperatorUnit end
@inline Base.hash(sector::Sector, h::UInt) = hash(id(sector), h)
@inline Base.:(==)(sector₁::Sector, sector₂::Sector) = isequal(id(sector₁), id(sector₂))
@inline Base.isequal(sector₁::Sector, sector₂::Sector) = isequal(id(sector₁), id(sector₂))

"""
    sumable(sector₁::Sector, sector₂::Sector) -> Bool

Judge whether two sectors could be direct summed.
"""
function sumable end

"""
    productable(sector₁::Sector, sector₂::Sector) -> Bool

Judge whether two sectors could be direct producted.
"""
function productable end

"""
    matrix(ops::Operators, braket::NTuple{2, Sector}, table, dtype=valtype(eltype(ops))) -> SparseMatrixCSC{dtype, Int}

Get the CSC-formed sparse matrix representation of a set of operators.

Here, `table` specifies the order of the operator ids.
"""
function matrix(ops::Operators, braket::NTuple{2, Sector}, table, dtype=valtype(eltype(ops)))
    length(ops)==0 && return spzeros(dtype, length(braket[1]), length(braket[2]))
    length(ops)==1 && return matrix(ops[1], braket, table, dtype)
    return matrix(ops[1:length(ops)÷2], braket, table, dtype) + matrix(ops[length(ops)÷2+1:length(ops)], braket, table, dtype)
end

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
    @assert all(map(previous->sumable(previous, sector), target.sectors)) "add! error: could not be direct summed."
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
    TargetSpace(hilbert::Hilbert)
    TargetSpace(hilbert::Hilbert, quantumnumbers::Union{AbelianNumber, Tuple{AbelianNumber, Vararg{AbelianNumber}}})

Construct a target space from the total Hilbert space and the associated quantum numbers.
"""
@inline TargetSpace(hilbert::Hilbert) = TargetSpace(Sector(hilbert))
@inline TargetSpace(hilbert::Hilbert, quantumnumbers::Union{AbelianNumber, Tuple{AbelianNumber, Vararg{AbelianNumber}}}) = TargetSpace(map(quantumnumber->Sector(hilbert, quantumnumber), wrapper(quantumnumbers))...)

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
    EDMatrix(braket::NTuple{2, Sector}, m::SparseMatrixCSC)

Construct a matrix representation when
1) the ket and bra spaces share the same bases;
2-3) the ket and bra spaces may be different.
"""
@inline EDMatrix(sector::Sector, m::SparseMatrixCSC) = EDMatrix(sector, sector, m)
@inline EDMatrix(braket::NTuple{2, Sector}, m::SparseMatrixCSC) = EDMatrix(braket[1], braket[2], m)

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
    eigen(m::EDMatrix; nev=6, which=:SR, tol=0.0, maxiter=300, sigma=nothing, v₀=dtype(m)[]) -> Eigen

Solve the eigen problem by the restarted Lanczos method provided by the Arpack package.
"""
@inline function eigen(m::EDMatrix; nev=6, which=:SR, tol=0.0, maxiter=300, sigma=nothing, v₀=dtype(m)[])
    @assert m.bra==m.ket "eigen error: eigen decomposition of an `EDMatrix` are only available for those with the same bra and ket spaces."
    if size(m.matrix)[1] > 1
        eigvals, eigvecs = eigs(m.matrix; nev=nev, which=which, tol=tol, maxiter=maxiter, sigma=sigma, ritzvec=true, v0=v₀)
        @assert norm(imag(eigvals))<10^-14 "eigen error: non-vanishing imaginary parts of the eigen values."
        eigvals = real(eigvals)
    else
        eigvals, eigvecs = eigen(collect(m.matrix))
    end
    return Eigen(eigvals, eigvecs)
end

"""
    eigen(ms::OperatorSum{<:EDMatrix}; nev::Int=1, tol::Real=0.0, maxiter::Int=300, v₀::Union{AbstractVector, Dict{<:Sector, <:AbstractVector}, Dict{<:AbelianNumber, <:AbstractVector}}=dtype(eltype(ms))[], timer::TimerOutput=edtimer)

Solve the eigen problem by the restarted Lanczos method provided by the Arpack package.
"""
@inline function eigen(ms::OperatorSum{<:EDMatrix}; nev::Int=1, tol::Real=0.0, maxiter::Int=300, v₀::Union{AbstractVector, Dict{<:Sector, <:AbstractVector}, Dict{<:AbelianNumber, <:AbstractVector}}=dtype(eltype(ms))[], timer::TimerOutput=edtimer)
    @timeit timer "eigen" begin
        isa(v₀, AbstractVector) && (v₀ = Dict(m.ket=>v₀ for m in ms))
        isa(v₀, Dict{<:AbelianNumber, <:AbstractVector}) && (v₀ = Dict(m.ket=>get(v₀, AbelianNumber(m.ket), dtype(eltype(ms))[]) for m in ms))
        values, vectors, sectors = real(dtype(eltype(ms)))[], Vector{dtype(eltype(ms))}[], eltype(idtype(eltype(ms)))[]
        for m in ms
            @timeit timer string(AbelianNumber(m.ket)) begin
                k = min(length(m.ket), nev)
                eigensystem = eigen(m; nev=k, which=:SR, tol=tol, maxiter=maxiter, v₀=get(v₀, m.ket, dtype(eltype(ms))[]))
                for i = 1:k
                    push!(values, eigensystem.values[i])
                    push!(vectors, eigensystem.vectors[:, i])
                    push!(sectors, m.ket)
                end
            end
        end
        perm = sortperm(values)[1:min(nev, length(values))]
        result = EDEigen(values[perm], vectors[perm], sectors[perm])
    end
    return result
end

"""
    EDMatrixRepresentation{D<:Number, S<:Sector, T} <: MatrixRepresentation

Exact matrix representation of a quantum lattice system on a target Hilbert space.
"""
struct EDMatrixRepresentation{D<:Number, S<:Sector, T} <: MatrixRepresentation
    brakets::Vector{Tuple{S, S}}
    table::T
    EDMatrixRepresentation{D}(brakets::Vector{Tuple{S, S}}, table) where {D<:Number, S<:Sector} = new{D, S, typeof(table)}(brakets, table)
end
@inline function Base.valtype(::Type{<:EDMatrixRepresentation{D, S}}, M::Type{<:Operator}) where {D<:Number, S<:Sector}
    @assert promote_type(D, valtype(M))==D "valtype error: convert $(valtype(M)) to $D is inexact."
    E = EDMatrix{S, SparseMatrixCSC{D, Int}}
    return OperatorSum{E, idtype(E)}
end
@inline Base.valtype(R::Type{<:EDMatrixRepresentation}, M::Type{<:Operators}) = valtype(R, eltype(M))
function (representation::EDMatrixRepresentation)(m::Union{Operator, Operators}; kwargs...)
    result = zero(valtype(representation, m))
    if isa(m, Operator) || length(m)>0
        for braket in representation.brakets
            add!(result, EDMatrix(braket, matrix(m, braket, representation.table, dtype(eltype(result)); kwargs...)))
        end
    end
    return result
end

"""
    EDMatrixRepresentation{D}(target::TargetSpace, table) where {D<:Number}

Construct a exact matrix representation.
"""
@inline EDMatrixRepresentation{D}(target::TargetSpace, table) where {D<:Number} = EDMatrixRepresentation{D}([(sector, sector) for sector in target], table)

"""
    SectorFilter{S} <: LinearTransformation

Filter the target bra and ket Hilbert spaces.
"""
struct SectorFilter{S} <: LinearTransformation
    brakets::Set{Tuple{S, S}}
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
    function ED{K}(lattice::AbstractLattice, H::OperatorGenerator, mr::EDMatrixRepresentation; timer::TimerOutput=edtimer, delay::Bool=false) where {K<:EDKind}
        @timeit timer "matrix" begin
            @timeit timer "prepare" Hₘ = delay ? Image(mr(empty(Entry(H))), mr, objectid(H)) : mr(H)
        end
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
    prepare!(ed::ED; timer::TimerOutput=edtimer) -> ED
    prepare!(ed::Algorithm{<:ED}) -> Algorithm{<:ED}

Prepare the matrix representation.
"""
@inline function prepare!(ed::ED; timer::TimerOutput=edtimer)
    @timeit timer "prepare" (isempty(ed.Hₘ) && reset!(ed.Hₘ, ed.Hₘ.transformation, ed.H))
    return ed
end
@inline prepare!(ed::Algorithm{<:ED}) = prepare!(ed.frontend; timer=ed.timer)

"""
    release!(ed::ED; gc::Bool=true) -> ED
    release!(ed::Algorithm{<:ED}; gc::Bool=true) -> Algorithm{<:ED}

Release the memory source used in preparing the matrix representation. If `gc` is `true`, call the garbage collection immediately.
"""
@inline function release!(ed::ED; gc::Bool=true)
    empty!(ed.Hₘ)
    gc && GC.gc()
    return ed
end
@inline release!(ed::Algorithm{<:ED}; gc::Bool=true) = release!(ed.frontend; gc=gc)

"""
    matrix(ed::ED, sectors::Union{AbelianNumber, Sector}...; timer::TimerOutput=edtimer, kwargs...) -> OperatorSum{<:EDMatrix}
    matrix(ed::Algorithm{<:ED}, sectors::Union{AbelianNumber, Sector}...; kwargs...) -> OperatorSum{<:EDMatrix}

Get the sparse matrix representation of a quantum lattice system in the target space.
"""
function matrix(ed::ED; timer::TimerOutput=edtimer, kwargs...)
    isempty(ed.Hₘ) && @warn("Empty matrix found. You may need to call `prepare!` first.")
    @timeit timer "matrix" begin
        @timeit timer "expand" (result = expand(ed.Hₘ))
    end
    return result
end
function matrix(ed::ED, sectors::Sector...; timer::TimerOutput=edtimer, kwargs...)
    isempty(ed.Hₘ) && @warn("Empty matrix found. You may need to call `prepare!` first.")
    @timeit timer "matrix" begin
        @timeit timer "expand" (result = expand(SectorFilter(sectors...)(ed.Hₘ)))
    end
    return result
end
function matrix(ed::ED, quantumnumbers::AbelianNumber...; timer::TimerOutput=edtimer, kwargs...)
    sectors = [braket[1] for braket in ed.Hₘ.transformation.brakets if AbelianNumber(braket[1]) in quantumnumbers]
    return matrix(ed, sectors...; timer=timer, kwargs...)
end
function matrix(ed::Algorithm{<:ED}, sectors::Union{AbelianNumber, Sector}...; kwargs...)
    return matrix(ed.frontend, sectors...; timer=ed.timer, kwargs...)
end

"""
    eigen(ed::ED, sectors::Union{AbelianNumber, Sector}...; timer::TimerOutput=edtimer, kwargs...) -> EDEigen
    eigen(ed::Algorithm{<:ED}, sectors::Union{AbelianNumber, Sector}...; kwargs...) -> EDEigen

Solve the eigen problem by the restarted Lanczos method provided by the Arpack package.
"""
@inline eigen(ed::ED, sectors::Union{AbelianNumber, Sector}...; timer::TimerOutput=edtimer, kwargs...) = eigen(matrix(ed, sectors...; timer=timer); timer=timer, kwargs...)
@inline eigen(ed::Algorithm{<:ED}, sectors::Union{AbelianNumber, Sector}...; kwargs...) = eigen(matrix(ed, sectors...; timer=ed.timer); timer=ed.timer, kwargs...)

"""
    ED(
        lattice::AbstractLattice, hilbert::Hilbert, terms::Union{Term, Tuple{Term, Vararg{Term}}}, targetspace::TargetSpace=TargetSpace(hilbert), dtype::Type{<:Number}=commontype(terms), boundary::Boundary=plain;
        neighbors::Union{Nothing, Int, Neighbors}=nothing, timer::TimerOutput=edtimer, delay::Bool=false
    )

Construct the exact diagonalization method for a quantum lattice system.
"""
function ED(
    lattice::AbstractLattice, hilbert::Hilbert, terms::Union{Term, Tuple{Term, Vararg{Term}}}, targetspace::TargetSpace=TargetSpace(hilbert), dtype::Type{<:Number}=commontype(terms), boundary::Boundary=plain;
    neighbors::Union{Nothing, Int, Neighbors}=nothing, timer::TimerOutput=edtimer, delay::Bool=false
)
    terms = wrapper(terms)
    isnothing(neighbors) && (neighbors = maximum(term->term.bondkind, terms))
    H = OperatorGenerator(terms, bonds(lattice, neighbors), hilbert, boundary, eager; half=false)
    mr = EDMatrixRepresentation{dtype}(targetspace, Table(hilbert, Metric(EDKind(hilbert), hilbert)))
    return ED{typeof(EDKind(hilbert))}(lattice, H, mr; timer=timer, delay=delay)
end
@inline wrapper(x) = (x,)
@inline wrapper(x::Tuple) = x
@inline commontype(term::Term) = valtype(term)
@inline @generated commontype(terms::Tuple{Vararg{Term}}) = mapreduce(valtype, promote_type, fieldtypes(terms))

"""
    ED(
        lattice::AbstractLattice, hilbert::Hilbert, terms::Union{Term, Tuple{Term, Vararg{Term}}}, quantumnumbers::Union{AbelianNumber, Tuple{AbelianNumber, Vararg{AbelianNumber}}}, dtype::Type{<:Number}=commontype(terms), boundary::Boundary=plain;
        neighbors::Union{Nothing, Int, Neighbors}=nothing, timer::TimerOutput=edtimer, delay::Bool=false
    )

Construct the exact diagonalization method for a quantum lattice system.
"""
function ED(
    lattice::AbstractLattice, hilbert::Hilbert, terms::Union{Term, Tuple{Term, Vararg{Term}}}, quantumnumbers::Union{AbelianNumber, Tuple{AbelianNumber, Vararg{AbelianNumber}}}, dtype::Type{<:Number}=commontype(terms), boundary::Boundary=plain;
    neighbors::Union{Nothing, Int, Neighbors}=nothing, timer::TimerOutput=edtimer, delay::Bool=false
)
    return ED(lattice, hilbert, terms, TargetSpace(hilbert, wrapper(quantumnumbers)), dtype, boundary; neighbors=neighbors, timer=timer, delay=delay)
end

end # module
