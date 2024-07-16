module CanonicalFockSystems

using Base.Iterators: product
using Printf: @printf
using QuantumLattices: iscreation, periods, rank, statistics
using QuantumLattices: AbelianNumber, Combinations, DuplicatePermutations, Fock, Hilbert, Index, Metric, Operator, Operators, OperatorUnitToTuple, ParticleNumber, SpinfulParticle, Table, VectorSpace
using SparseArrays: SparseMatrixCSC
using ..EDCore: ED, EDKind, EDMatrixRepresentation, Sector, TargetSpace, wrapper

import QuantumLattices: ⊗, id, matrix
import ..EDCore: productable, sumable

export BinaryBases, BinaryBasis, BinaryBasisRange, basistype

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

Construct a binary basis with the given occupied orbitals.
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
    BinaryBases{A<:AbelianNumber, B<:BinaryBasis, T<:AbstractVector{B}} <: Sector

A set of binary bases.
"""
struct BinaryBases{A<:AbelianNumber, B<:BinaryBasis, T<:AbstractVector{B}} <: Sector
    quantumnumbers::Vector{A}
    stategroups::Vector{B}
    table::T
end
@inline id(bs::BinaryBases) = (bs.quantumnumbers, bs.stategroups)
@inline Base.issorted(::BinaryBases) = true
@inline Base.length(bs::BinaryBases) = length(bs.table)
@inline Base.getindex(bs::BinaryBases, i::Integer) = bs.table[i]
@inline Base.eltype(bs::BinaryBases) = eltype(typeof(bs))
@inline Base.eltype(::Type{<:BinaryBases{<:AbelianNumber, B}}) where {B<:BinaryBasis} = B
@inline Base.iterate(bs::BinaryBases, state=1) = state>length(bs) ? nothing : (bs.table[state], state+1)
function Base.show(io::IO, bs::BinaryBases)
    for (i, (qn, group)) in enumerate(zip(bs.quantumnumbers, bs.stategroups))
        @printf io "{2^[%s]: %s}" join(collect(group), " ") qn
        i<length(bs.quantumnumbers) && @printf io "%s" " ⊗ "
    end
end
@inline Base.searchsortedfirst(b::BinaryBasis, bs::BinaryBases) = searchsortedfirst(bs.table, b)
function Base.show(io::IO, bs::BinaryBases{<:AbelianNumber, <:BinaryBasis, <:BinaryBasisRange})
    for (i, (qn, group)) in enumerate(zip(bs.quantumnumbers, bs.stategroups))
        @printf io "{2^1:%s}" maximum(collect(group))
        i<length(bs.quantumnumbers) && @printf io "%s" " ⊗ "
    end
end
@inline Base.searchsortedfirst(b::BinaryBasis, bs::BinaryBases{<:AbelianNumber, <:BinaryBasis, <:BinaryBasisRange}) = Int(b.rep+1)

"""
    BinaryBases(states)
    BinaryBases(nstate::Integer)
    BinaryBases{A}(states) where {A<:AbelianNumber}
    BinaryBases{A}(nstate::Integer) where {A<:AbelianNumber}

Construct a set of binary bases that subjects to no quantum number conservation.
"""
@inline BinaryBases(argument) = BinaryBases{ParticleNumber}(argument)
function BinaryBases{A}(nstate::Integer) where {A<:AbelianNumber}
    quantumnumber = A(map(p->NaN, periods(A))...)
    stategroup = BinaryBasis(one(nstate):nstate)
    table = BinaryBasisRange(nstate)
    return BinaryBases([quantumnumber], [stategroup], table)
end
function BinaryBases{A}(states) where {A<:AbelianNumber}
    quantumnumber = A(map(p->NaN, periods(A))...)
    stategroup = BinaryBasis(states)
    table = BinaryBasis{basistype(eltype(states))}[]
    table!(table, NTuple{length(states), basistype(eltype(states))}(sort!(collect(states); rev=true)))
    return BinaryBases([quantumnumber], [stategroup], table)
end
function table!(table, states::NTuple{N}) where N
    for poses in DuplicatePermutations{N}((false, true))
        push!(table, BinaryBasis(states; filter=index->poses[index]))
    end
    return table
end

"""
    BinaryBases(states, nparticle::Integer)
    BinaryBases(nstate::Integer, nparticle::Integer)
    BinaryBases{A}(states, nparticle::Integer; kwargs...) where {A<:AbelianNumber}
    BinaryBases{A}(nstate::Integer, nparticle::Integer; kwargs...) where {A<:AbelianNumber}

Construct a set of binary bases that preserves the particle number conservation.
"""
@inline BinaryBases(argument, nparticle::Integer) = BinaryBases{ParticleNumber}(argument, nparticle)
@inline BinaryBases{A}(nstate::Integer, nparticle::Integer; kwargs...) where {A<:AbelianNumber} = BinaryBases{A}(one(nstate):nstate, nparticle; kwargs...)
function BinaryBases{A}(states, nparticle::Integer; kwargs...) where {A<:AbelianNumber}
    kwargs = (kwargs..., N=nparticle)
    quantumnumber = A(map(fieldname->getfield(kwargs, fieldname), fieldnames(A))...)
    stategroup = BinaryBasis(states)
    table = BinaryBasis{basistype(eltype(states))}[]
    table!(table, NTuple{length(states), basistype(eltype(states))}(sort!(collect(states); rev=true)), Val(nparticle))
    return BinaryBases([quantumnumber], [stategroup], table)
end
function table!(table, states::Tuple, ::Val{N}) where N
    for poses in Combinations{N}(states)
        push!(table, BinaryBasis{eltype(states)}(poses))
    end
    return reverse!(table)
end

"""
    BinaryBases(spindws, spinups, sz::Real)
    BinaryBases{A}(spindws, spinups, sz::Real; kwargs...) where {A<:AbelianNumber}

Construct a set of binary bases that preserves the spin z component but not the particle number conservation.
"""
@inline BinaryBases(spindws, spinups, sz::Real) = BinaryBases{SpinfulParticle}(spindws, spinups, sz)
function BinaryBases{A}(spindws, spinups, sz::Real; kwargs...) where {A<:AbelianNumber}
    kwargs = (kwargs..., N=NaN, Sz=sz)
    quantumnumber = A(map(fieldname->getfield(kwargs, fieldname), fieldnames(A))...)
    stategroup = BinaryBasis([spindws..., spinups...])
    basistable = typeof(stategroup)[]
    for nup in max(Int(2*quantumnumber.Sz), 0):min(length(spinups)+Int(2*quantumnumber.Sz), length(spinups))
        ndw = nup-Int(2*quantumnumber.Sz)
        append!(basistable, BinaryBases(spindws, ndw) ⊗ BinaryBases(spinups, nup))
    end
    return BinaryBases([quantumnumber], [stategroup], sort!(basistable)::Vector{typeof(stategroup)})
end

"""
    BinaryBases(spindws, spinups, nparticle::Integer, sz::Real)
    BinaryBases{A}(spindws, spinups, nparticle::Integer, sz::Real; kwargs...) where {A<:AbelianNumber}

Construct a set of binary bases that preserves both the particle number and the spin z component conservation.
"""
@inline BinaryBases(spindws, spinups, nparticle::Integer, sz::Real) = BinaryBases{SpinfulParticle}(spindws, spinups, nparticle, sz)
function BinaryBases{A}(spindws, spinups, nparticle::Integer, sz::Real; kwargs...) where {A<:AbelianNumber}
    kwargs = (kwargs..., N=nparticle, Sz=sz)
    quantumnumber = A(map(fieldname->getfield(kwargs, fieldname), fieldnames(A))...)
    ndw, nup = Int(quantumnumber.N/2-quantumnumber.Sz), Int(quantumnumber.N/2+quantumnumber.Sz)
    return BinaryBases{SpinfulParticle}(spindws, ndw; Sz=-0.5*ndw) ⊗ BinaryBases{SpinfulParticle}(spinups, nup; Sz=0.5*nup)
end

"""
    AbelianNumber(bs::BinaryBases)

Get the Abelian quantum number of a set of binary bases.
"""
@inline AbelianNumber(bs::BinaryBases) = sum(bs.quantumnumbers)

"""
    sumable(bs₁::BinaryBases, bs₂::BinaryBases) -> Bool

Judge whether two sets of binary bases could be direct summed.
"""
function sumable(bs₁::BinaryBases{A₁}, bs₂::BinaryBases{A₂}) where {A₁<:AbelianNumber, A₂<:AbelianNumber}
    # A₁==A₂ || return false
    # AbelianNumber(bs₁)==AbelianNumber(bs₂) || return true
    true
end

"""
    productable(bs₁::BinaryBases, bs₂::BinaryBases) -> Bool

Judge whether two sets of binary bases could be direct producted.
"""
function productable(bs₁::BinaryBases{A₁}, bs₂::BinaryBases{A₂}) where {A₁<:AbelianNumber, A₂<:AbelianNumber}
    A₁==A₂ || return false
    for (group₁, group₂) in product(bs₁.stategroups, bs₂.stategroups)
        isequal(group₁.rep & group₂.rep, 0) || return false
    end
    return true
end

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
    EDKind(::Type{<:Hilbert{<:Fock}})

The kind of the exact diagonalization method applied to a canonical quantum Fock lattice system.
"""
@inline EDKind(::Type{<:Hilbert{<:Fock}}) = EDKind(:FED)

"""
    Metric(::EDKind{:FED}, ::Hilbert{<:Fock}) -> OperatorUnitToTuple

Get the index-to-tuple metric for a canonical quantum Fock lattice system.
"""
@inline @generated Metric(::EDKind{:FED}, ::Hilbert{<:Fock}) = OperatorUnitToTuple(:spin, :site, :orbital)

"""
    Sector(hilbert::Hilbert{<:Fock}, basistype=UInt) -> BinaryBases
    Sector(hilbert::Hilbert{<:Fock}, quantumnumber::ParticleNumber, basistype=UInt; table=Table(hilbert, Metric(EDKind(hilbert), hilbert))) -> BinaryBases
    Sector(hilbert::Hilbert{<:Fock}, quantumnumber::SpinfulParticle, basistype=UInt; table=Table(hilbert, Metric(EDKind(hilbert), hilbert))) -> BinaryBases

Construct the binary bases of a Hilbert space with the specified quantum number.
"""
@inline Sector(hilbert::Hilbert{<:Fock}, basistype=UInt) = BinaryBases(basistype(sum([length(internal)÷2 for internal in values(hilbert)])))
function Sector(hilbert::Hilbert{<:Fock}, quantumnumber::ParticleNumber, basistype=UInt; table=Table(hilbert, Metric(EDKind(hilbert), hilbert)))
    states = Set{basistype}(table[Index(site, iid)] for (site, internal) in hilbert for iid in internal)
    if isnan(quantumnumber.N)
        return BinaryBases{ParticleNumber}(states)
    else
        return BinaryBases{ParticleNumber}(states, Int(quantumnumber.N))
    end
end
function Sector(hilbert::Hilbert{<:Fock}, quantumnumber::SpinfulParticle, basistype=UInt; table=Table(hilbert, Metric(EDKind(hilbert), hilbert)))
    @assert all(internal->internal.nspin==2, values(hilbert)) "Sector error: only for spin-1/2 systems."
    if isnan(quantumnumber.Sz)
        states = Set{basistype}(table[Index(site, iid)] for (site, internal) in hilbert for iid in internal)
        if isnan(quantumnumber.N)
            return BinaryBases{SpinfulParticle}(states)
        else
            return BinaryBases{SpinfulParticle}(states, Int(quantumnumber.N); Sz=NaN)
        end
    else
        spindws = Set{basistype}(table[Index(site, iid)] for (site, internal) in hilbert for iid in internal if iid.spin==-1//2)
        spinups = Set{basistype}(table[Index(site, iid)] for (site, internal) in hilbert for iid in internal if iid.spin==+1//2)
        if isnan(quantumnumber.N)
            return BinaryBases{SpinfulParticle}(spindws, spinups, quantumnumber.Sz)
        else
            return BinaryBases{SpinfulParticle}(spindws, spinups, Int(quantumnumber.N), quantumnumber.Sz)
        end
    end
end

"""
    TargetSpace(hilbert::Hilbert{<:Fock}, basistype=UInt)
    TargetSpace(hilbert::Hilbert{<:Fock}, quantumnumbers::Union{AbelianNumber, Tuple{AbelianNumber, Vararg{AbelianNumber}}}, basistype=UInt; table=Table(hilbert, Metric(EDKind(hilbert), hilbert)))

Construct a target space from the total Hilbert space and the associated quantum numbers.
"""
@inline TargetSpace(hilbert::Hilbert{<:Fock}, basistype=UInt) = TargetSpace(Sector(hilbert, basistype))
@inline function TargetSpace(hilbert::Hilbert{<:Fock}, quantumnumbers::Union{AbelianNumber, Tuple{AbelianNumber, Vararg{AbelianNumber}}}, basistype=UInt; table=Table(hilbert, Metric(EDKind(hilbert), hilbert)))
    return TargetSpace(map(quantumnumber->Sector(hilbert, quantumnumber, basistype; table=table), wrapper(quantumnumbers))...)
end

"""
    matrix(op::Operator, braket::NTuple{2, BinaryBases}, table, dtype=valtype(op)) -> SparseMatrixCSC{dtype, Int}

Get the CSC-formed sparse matrix representation of an operator.

Here, `table` specifies the order of the operator ids.
"""
function matrix(op::Operator, braket::NTuple{2, BinaryBases}, table, dtype=valtype(op))
    bra, ket = braket[1], braket[2]
    @assert bra.stategroups==ket.stategroups "matrix error: mismatched bra and ket."
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

end # module
