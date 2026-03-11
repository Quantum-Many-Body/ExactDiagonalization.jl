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

Construct a set of binary bases that subject to no quantum number conservation.
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
    @assert productable(bs₁, bs₂) "⊗ error: the input two sets of bases cannot form a direct product."
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
    matrix(op::Operator{V, <:OneAtLeast{OperatorIndex}}, braket::NTuple{2, BinaryBases}, table::AbstractDict, dtype=V) where V -> SparseMatrixCSC{dtype, Int}

Get the CSC-formed sparse matrix representation of an operator.

Here, `table` specifies the order of the operator indexes.
"""
function matrix(op::Operator{V, <:OneAtLeast{OperatorIndex}}, braket::NTuple{2, BinaryBases}, table::AbstractDict, dtype=V) where V
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
