module CanonicalFockSystems

using Base.Iterators: product
using Printf: @printf, @sprintf
using QuantumLattices: plain, bonds, id, iscreation, periods, rank
using QuantumLattices: AbstractLattice, AbelianNumber, Boundary, Combinations, DuplicatePermutations, Hilbert, Fock, FockTerm, Metric, Neighbors, Operator, OperatorGenerator, Operators, OperatorUnitToTuple, ParticleNumber, Table, Term, VectorSpace
using SparseArrays: SparseMatrixCSC, spzeros
using ..EDCore: ED, EDKind, EDMatrixRepresentation, Sector, TargetSpace

import QuantumLattices: ⊗, matrix, statistics

export BinaryBases, BinaryBasis, BinaryBasisRange, productable, sumable

# Binary bases commonly used in canonical fermionic and hardcore bosonic quantum lattice systems
"""
    BinaryBasis{I<:Unsigned}

Binary basis represented by an unsigned integer.
"""
struct BinaryBasis{I<:Unsigned}
    rep::I
    BinaryBasis(i::Integer) = (rep = Unsigned(i); new{typeof(rep)}(rep))
end
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
    iterate(basis::BinaryBasis, state=nothing)

Iterate over the numbers of the occupied single-particle orbitals.
"""
function Base.iterate(basis::BinaryBasis, state=nothing)
    (pos, rep) = isnothing(state) ? (0, basis.rep) : (state[1], state[2])
    while rep>0
        pos += 1
        isodd(rep) && return (pos, (pos, rep÷2))
        rep ÷= 2
    end
    return nothing
end

"""
    one(basis::BinaryBasis, state::Integer) -> BinaryBasis

Get a new basis with the specified single-particle state occupied. 
"""
@inline Base.one(basis::BinaryBasis, state::Integer) = BinaryBasis(basis.rep | one(basis.rep)<<(state-1))

"""
    isone(basis::BinaryBasis, state::Integer) -> Bool

Judge whether the specified single-particle state is occupied for a basis.
"""
@inline Base.isone(basis::BinaryBasis, state::Integer) = (basis.rep & one(basis.rep)<<(state-1))>0

"""
    zero(basis::BinaryBasis, state::Integer) -> BinaryBasis

Get a new basis with the specified single-particle state unoccupied.
"""
@inline Base.zero(basis::BinaryBasis, state::Integer) = BinaryBasis(basis.rep & ~(one(basis.rep)<<(state-1)))

"""
    iszero(basis::BinaryBasis, state::Integer) -> Bool

Judge whether the specified single-particle state is unoccupied for a basis.
"""
@inline Base.iszero(basis::BinaryBasis, state::Integer) = !isone(basis, state)

"""
    count(basis::BinaryBasis) -> Int
    count(basis::BinaryBasis, start::Integer, stop::Integer) -> Int

Count the number of occupied single-particle states.
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
@inline ⊗(basis₁::BinaryBasis, basis₂::BinaryBasis) = BinaryBasis(basis₁.rep|basis₂.rep)

"""
    BinaryBasis(states; filter=index->true)
    BinaryBasis{I}(states; filter=index->true) where {I<:Unsigned}

Construct a binary basis with the given occupied orbitals.
"""
@inline BinaryBasis(states; filter=index->true) = BinaryBasis{typeof(Unsigned(first(states)))}(states; filter=filter)
function BinaryBasis{I}(states; filter=index->true) where {I<:Unsigned}
    rep, eye = zero(I), one(I)
    for (index, state) in enumerate(states)
        filter(index) && (rep += eye<<(state-1))
    end
    return BinaryBasis(rep)
end

"""
    BinaryBasisRange{I<:Unsigned} <: VectorSpace{BinaryBasis{I}}

A continuous range of binary basis.
"""
struct BinaryBasisRange{I<:Unsigned} <: VectorSpace{BinaryBasis{I}}
    slice::UnitRange{I}
end
@inline Base.issorted(::BinaryBasisRange) = true
@inline Base.length(bbr::BinaryBasisRange) = length(bbr.slice)
@inline Base.getindex(bbr::BinaryBasisRange, i::Integer) = BinaryBasis(bbr.slice[i])

"""
    BinaryBases{A<:AbelianNumber, B<:BinaryBasis, T<:AbstractVector{B}} <: Sector

A set of binary bases.
"""
struct BinaryBases{A<:AbelianNumber, B<:BinaryBasis, T<:AbstractVector{B}} <: Sector
    id::Vector{Tuple{B, A}}
    table::T
end
@inline Base.issorted(::BinaryBases) = true
@inline Base.length(bs::BinaryBases) = length(bs.table)
@inline Base.:(==)(bs₁::BinaryBases, bs₂::BinaryBases) = isequal(bs₁.id, bs₂.id)
@inline Base.isequal(bs₁::BinaryBases, bs₂::BinaryBases) = isequal(bs₁.id, bs₂.id)
@inline Base.getindex(bs::BinaryBases, i::Integer) = bs.table[i]
@inline Base.eltype(bs::BinaryBases) = eltype(typeof(bs))
@inline Base.eltype(::Type{<:BinaryBases{<:AbelianNumber, B}}) where {B<:BinaryBasis} = B
@inline Base.iterate(bs::BinaryBases, state=1) = state>length(bs) ? nothing : (bs.table[state], state+1)
function Base.repr(bs::BinaryBases)
    result = String[]
    for (states, qn) in bs.id
        push!(result, @sprintf "{2^%s: %s}" count(states) qn)
    end
    return join(result, " ⊗ ")
end
function Base.show(io::IO, bs::BinaryBases)
    for (i, (states, qn)) in enumerate(bs.id)
        @printf io "{2^[%s]: %s}" join(collect(states), " ") qn
        i<length(bs.id) && @printf io "%s" " ⊗ "
    end
end
@inline Base.searchsortedfirst(b::BinaryBasis, bs::BinaryBases) = searchsortedfirst(bs.table, b)
@inline Base.searchsortedfirst(b::BinaryBasis, bs::BinaryBases{<:AbelianNumber, <:BinaryBasis, <:BinaryBasisRange}) = Int(b.rep+1)

"""
    AbelianNumber(bs::BinaryBases)

Get the Abelian quantum number of a set of binary bases.
"""
AbelianNumber(bs::BinaryBases) = sum(rep->rep[2], bs.id)

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
    return BinaryBases(sort!([bs₁.id; bs₂.id]; by=first), sort!(table))
end

"""
    productable(bs₁::BinaryBases, bs₂::BinaryBases) -> Bool

Judge whether two sets of binary bases could be direct producted.
"""
function productable(bs₁::BinaryBases{A₁}, bs₂::BinaryBases{A₂}) where {A₁, A₂}
    A₁==A₂ || return false
    for (irr₁, irr₂) in product(bs₁.id, bs₂.id)
        isequal(irr₁[1].rep & irr₂[1].rep, 0) || return false
    end
    return true
end

"""
    sumable(bs₁::BinaryBases, bs₂::BinaryBases) -> Bool

Judge whether two sets of binary bases could be direct summed.

Strictly speaking, two sets of binary bases could be direct summed if and only if they have no intersection. The time complexity to check the intersection is O(n log n), which costs a lot when the dimension of the binary bases is huge. It is also possible to judge whether they could be direct summed by close investigations on their ids, i.e. the single-particle states and occupation number. It turns out that this is a multi-variable pure integer linear programming problem. In the future, this function would be implemented based on this observation. At present, the direct summability should be handled by the users in priori.
"""
@inline sumable(bs₁::BinaryBases, bs₂::BinaryBases) = true

"""
    BinaryBases(nstate::Integer)
    BinaryBases(states)
    BinaryBases{A}(nstate::Integer) where {A<:AbelianNumber}
    BinaryBases{A}(states) where {A<:AbelianNumber}

Construct a set of binary bases that subject to no quantum number conservation.
"""
@inline BinaryBases(nstate::Integer) = BinaryBases{ParticleNumber}(nstate)
@inline BinaryBases{A}(nstate::Integer) where {A<:AbelianNumber} = BinaryBases([(BinaryBasis(1:nstate), A(map(p->NaN, periods(A))...))], BinaryBasisRange(UInt(0):UInt(2^nstate-1)))
@inline BinaryBases(states) = BinaryBases{ParticleNumber}(states)
@inline BinaryBases{A}(states) where {A<:AbelianNumber} = BinaryBases{A}(Tuple(states))
function BinaryBases{A}(states::NTuple{N, Integer}) where {A<:AbelianNumber, N}
    states = NTuple{N, eltype(states)}(sort!(collect(states); rev=true))
    com = DuplicatePermutations{N}((false, true))
    table = Vector{BinaryBasis{typeof(Unsigned(first(states)))}}(undef, length(com))
    for (i, poses) in enumerate(com)
        table[i] = BinaryBasis(states; filter=index->poses[index])
    end
    return BinaryBases([(BinaryBasis(states), A(map(p->NaN, periods(A))...))], table)
end

"""
    BinaryBases(nstate::Integer, nparticle::Integer)
    BinaryBases(states, nparticle::Integer)
    BinaryBases{A}(nstate::Integer, nparticle::Integer; kwargs...) where {A<:AbelianNumber}
    BinaryBases{A}(states, nparticle::Integer; kwargs...) where {A<:AbelianNumber}

Construct a set of binary bases that preserves the particle number conservation.
"""
@inline BinaryBases(nstate::Integer, nparticle::Integer) = BinaryBases{ParticleNumber}(nstate, nparticle)
@inline BinaryBases(states, nparticle::Integer) = BinaryBases{ParticleNumber}(states, nparticle)
@inline BinaryBases{A}(nstate::Integer, nparticle::Integer; kwargs...) where {A<:AbelianNumber} = BinaryBases{A}(1:nstate, Val(nparticle); kwargs...)
@inline BinaryBases{A}(states, nparticle::Integer; kwargs...) where {A<:AbelianNumber} = BinaryBases{A}(states, Val(nparticle); kwargs...)
function BinaryBases{A}(states, ::Val{N}; kwargs...) where {A<:AbelianNumber, N}
    com = Combinations{N}(sort!(collect(states); rev=true))
    I = typeof(Unsigned(first(states)))
    table = Vector{BinaryBasis{I}}(undef, length(com))
    for (i, poses) in enumerate(com)
        table[end+1-i] = BinaryBasis{I}(poses)
    end
    kwargs = (kwargs..., N=N)
    return BinaryBases([(BinaryBasis{I}(states), A(map(fieldname->getfield(kwargs, fieldname), fieldnames(A))...))], table)
end

# CSC-formed sparse matrix representation of an operator
"""
    matrix(op::Operator, braket::NTuple{2, BinaryBases}, table; dtype=valtype(op)) -> SparseMatrixCSC{dtype, Int}
    matrix(ops::Operators, braket::NTuple{2, BinaryBases}, table; dtype=valtype(eltype(ops))) -> SparseMatrixCSC{dtype, Int}

Get the CSC-formed sparse matrix representation of an operator.

Here, `table` specifies the order of the operator ids.
"""
function matrix(op::Operator, braket::NTuple{2, BinaryBases}, table; dtype=valtype(op))
    bra, ket = braket[1], braket[2]
    ndata, intermediate = 1, zeros(ket|>eltype, rank(op)+1)
    data, indices, indptr = zeros(dtype, length(ket)), zeros(Int, length(ket)), zeros(Int, length(ket)+1)
    sequences = NTuple{rank(op), Int}(table[op[i]] for i in reverse(1:rank(op)))
    iscreations = NTuple{rank(op), Bool}(iscreation(index) for index in reverse(id(op)))
    for i = 1:length(ket)
        flag = true
        indptr[i] = ndata
        intermediate[1] = ket[i]
        for j = 1:rank(op)
            isone(intermediate[j], sequences[j])==iscreations[j] && (flag = false; break)
            intermediate[j+1] = iscreations[j] ? one(intermediate[j], sequences[j]) : zero(intermediate[j], sequences[j])
        end
        if flag
            nsign = 0
            statistics(eltype(op))==:f && for j = 1:rank(op)
                nsign += count(intermediate[j], 1, sequences[j]-1)
            end
            index = searchsortedfirst(intermediate[end], bra)
            if index<=length(bra) && bra[index]==intermediate[end]
                indices[ndata] = index
                data[ndata] = op.value*(-1)^nsign
                ndata += 1
            end
        end
    end
    indptr[end] = ndata
    return SparseMatrixCSC(length(bra), length(ket), indptr, indices[1:ndata-1], data[1:ndata-1])
end
function matrix(ops::Operators, braket::NTuple{2, BinaryBases}, table; dtype=valtype(eltype(ops)))
    result = spzeros(dtype, length(braket[1]), length(braket[2]))
    for op in ops
        result += matrix(op, braket, table; dtype=dtype)
    end
    return result
end

"""
    EDKind(::Type{<:FockTerm})

The kind of the exact diagonalization method applied to a canonical quantum Fock lattice system.
"""
@inline EDKind(::Type{<:FockTerm}) = EDKind(:FED)

"""
    Metric(::EDKind{:FED}, ::Hilbert{<:Fock}) -> OperatorUnitToTuple

Get the index-to-tuple metric for a canonical quantum Fock lattice system.
"""
@inline @generated Metric(::EDKind{:FED}, ::Hilbert{<:Fock}) = OperatorUnitToTuple(:spin, :site, :orbital)

"""
    ED(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}, targetspace::TargetSpace; neighbors::Union{Nothing, Int, Neighbors}=nothing, boundary::Boundary=plain)

Construct the exact diagonalization method for a canonical quantum Fock lattice system.
"""
function ED(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}, targetspace::TargetSpace; neighbors::Union{Nothing, Int, Neighbors}=nothing, boundary::Boundary=plain)
    k = EDKind(typeof(terms))
    isnothing(neighbors) && (neighbors = maximum(term->term.bondkind, terms))
    H = OperatorGenerator(terms, bonds(lattice, neighbors), hilbert; half=false, boundary=boundary)
    mr = EDMatrixRepresentation(targetspace, Table(hilbert, Metric(k, hilbert)))
    return ED{typeof(k)}(lattice, H, mr)
end

end # module
