module CanonicalSpinSystems

using Base.Iterators: product
using LinearAlgebra: I
using LuxurySparse: SparseMatrixCOO
using Printf: @printf
using QuantumLattices: findindex, totalspin
using QuantumLattices: AbelianNumber, AbelianNumbers, Hilbert, Metric, Operator, OperatorUnitToTuple, Spin, Sz
using SparseArrays: SparseMatrixCSC, nnz, nonzeros, nzrange, rowvals, sparse
using ..EDCore: EDKind, Sector, TargetSpace, wrapper

import QuantumLattices: id, matrix
import ..EDCore: sumable

export SpinBases

struct SpinSlice{P<:Union{Vector{Int}, Colon}}
    positions::Vector{Int}
    quantumnumbers::AbelianNumbers{Sz}
    permutation::P
end
"""
    SpinBases <: Sector

A set of spin bases.
"""
struct SpinBases{N} <: Sector
    quantumnumber::Sz
    spins::Vector{Rational{Int}}
    partition::NTuple{N, SpinSlice}
    record::Dict{NTuple{N, Sz}, UnitRange{Int}}
    function SpinBases(quantumnumber::Sz, spins::Vector{Rational{Int}}, partition::NTuple{N, SpinSlice}, record::Dict{NTuple{N, Sz}, UnitRange{Int}}) where N
        @assert Set(vcat(map(slice->slice.positions, partition)...))==Set(1:length(spins)) "SpinBases error: incorrect partition."
        new{N}(quantumnumber, spins, partition, record)
    end
end
@inline id(bs::SpinBases) = (bs.quantumnumber, bs.spins, map(slice->slice.positions, bs.partition))
@inline Base.length(bs::SpinBases) = maximum(range->maximum(range), values(bs.record))
function Base.show(io::IO, bs::SpinBases)
    tostr(spin::Rational{Int}, order::Int) = string(spin.den==1 ? string(spin.num) : string(spin.num, "/", spin.den), join('₀'+d for d in reverse(digits(order))))
    @printf io "%s" "{"
    for (i, slice) in enumerate(bs.partition)
        @printf io "(%s)" join([tostr(bs.spins[position], position) for position in slice.positions], "⊗")
        i<length(bs.partition) && @printf io "%s" " ⊗ "
    end
    @printf io ": %s}" bs.quantumnumber
end
function Base.match(bs₁::SpinBases{N₁}, bs₂::SpinBases{N₂}) where {N₁, N₂}
    N₁==N₂ || return false
    isnan(bs₁.quantumnumber.Sz)==isnan(bs₂.quantumnumber.Sz) || return false
    bs₁.spins==bs₂.spins || return false
    for (slice₁, slice₂) in zip(bs₁.partition, bs₂.partition)
        slice₁.positions==slice₂.positions || return false
    end
    return true
end

"""
    SpinBases(spins::Vector{<:Real})
    SpinBases(spins::Vector{<:Real}, partition::NTuple{N, AbstractVector{Int}}) where N
    SpinBases(spins::Vector{<:Real}, quantumnumber::Sz)
    SpinBases(spins::Vector{<:Real}, quantumnumber::Sz, partition::NTuple{N, AbstractVector{Int}}) where N

Construct a set of spin bases.
"""
@inline SpinBases(spins::Vector{<:Real}) = SpinBases(spins, defaultpartition(length(spins)))
function SpinBases(spins::Vector{<:Real}, partition::NTuple{N, AbstractVector{Int}}) where N
    spins = collect(Rational{Int}, spins)
    @assert all(spin->spin.den∈(1, 2), spins) "SpinBases error: incorrect input spins."
    partition = map(partition) do positions
        SpinSlice(positions, AbelianNumbers('c', [Sz(NaN)], [prod(map(position->Int(2*spins[position]+1), positions))], :counts), :)
    end
    record = Dict(ntuple(i->Sz(NaN), Val(N))=>1:prod(map(spin->Int(2*spin+1), spins)))
    return SpinBases(Sz(NaN), spins, partition, record)
end
@inline SpinBases(spins::Vector{<:Real}, quantumnumber::Sz) = SpinBases(spins, quantumnumber, defaultpartition(length(spins)))
function SpinBases(spins::Vector{<:Real}, quantumnumber::Sz, partition::NTuple{N, AbstractVector{Int}}) where N
    @assert !isnan(quantumnumber.Sz) "SpinBases error: when Sz is not conserved, no quantum number should be used."
    spins = collect(Rational{Int}, spins)
    @assert all(spin->spin.den∈(1, 2), spins) "SpinBases error: incorrect input spins."
    quantumnumberses, permutations, records = intermediate(spins, partition)
    partition = ntuple(i->SpinSlice(partition[i], quantumnumberses[i], permutations[i]), Val(N))
    return SpinBases(quantumnumber, spins, partition, records[quantumnumber])
end
@inline defaultpartition(n::Int) =(cut=n÷2; (collect(1:cut), collect(cut+1:n)))
function intermediate(spins, partition)
    quantumnumberses, permutations = AbelianNumbers{Sz}[], Vector{Int}[]
    for positions in partition
        if length(positions)>0
            quantumnumbers, permutation = sort(kron([spinzs(spins[position]) for position in positions]...))
            push!(quantumnumberses, quantumnumbers)
            push!(permutations, permutation)
        end
    end
    return quantumnumberses, permutations, prod(quantumnumberses...)[2]
end
@inline spinzs(S::Real) = AbelianNumbers('U', [Sz(sz) for sz = S:-1:-S], collect(0:Int(2*S+1)), :indptr)

"""
    AbelianNumber(bs::SpinBases)

Get the quantum number of a set of spin bases.
"""
@inline AbelianNumber(bs::SpinBases) = bs.quantumnumber

"""
    sumable(bs₁::SpinBases, bs₂::SpinBases) -> Bool

Judge whether two sets of spin bases could be direct summed.
"""
@inline sumable(bs₁::SpinBases, bs₂::SpinBases) = AbelianNumber(bs₁) ≠ AbelianNumber(bs₂)

"""
    EDKind(::Type{<:Hilbert{<:Spin}})

The kind of the exact diagonalization method applied to a canonical quantum spin lattice system.
"""
@inline EDKind(::Type{<:Hilbert{<:Spin}}) = EDKind(:SED)

"""
    Metric(::EDKind{:SED}, ::Hilbert{<:Spin}) -> OperatorUnitToTuple

Get the index-to-tuple metric for a canonical quantum spin lattice system.
"""
@inline @generated Metric(::EDKind{:SED}, ::Hilbert{<:Spin}) = OperatorUnitToTuple(:site)

"""
    Sector(hilbert::Hilbert{<:Spin}, partition::Tuple{Vararg{AbstractVector{Int}}}=defaultpartition(length(hilbert)))
    Sector(hilbert::Hilbert{<:Spin}, quantumnumber::Sz, partition::Tuple{Vararg{AbstractVector{Int}}}=defaultpartition(length(hilbert)))

Construct the spin bases of a Hilbert space with the specified quantum number.
"""
@inline Sector(hilbert::Hilbert{<:Spin}, partition::Tuple{Vararg{AbstractVector{Int}}}=defaultpartition(length(hilbert))) = SpinBases(sorted_spins(hilbert), partition)
@inline Sector(hilbert::Hilbert{<:Spin}, quantumnumber::Sz, partition::Tuple{Vararg{AbstractVector{Int}}}=defaultpartition(length(hilbert))) = SpinBases(sorted_spins(hilbert), quantumnumber, partition)
@inline sorted_spins(hilbert::Hilbert{<:Spin}) = permute!([convert(Rational{Int}, totalspin(spin)) for spin in values(hilbert)], sortperm(collect(keys(hilbert))))

"""
    TargetSpace(hilbert::Hilbert{<:Spin}, partition::Tuple{Vararg{AbstractVector{Int}}}=defaultpartition(length(hilbert)))
    TargetSpace(hilbert::Hilbert{<:Spin}, quantumnumbers::Union{Sz, Tuple{Sz, Vararg{Sz}}}, partition::Tuple{Vararg{AbstractVector{Int}}}=defaultpartition(length(hilbert)))

Construct a target space from the total Hilbert space and the associated quantum numbers.
"""
@inline TargetSpace(hilbert::Hilbert{<:Spin}, partition::Tuple{Vararg{AbstractVector{Int}}}=defaultpartition(length(hilbert))) = TargetSpace(Sector(hilbert, partition))
function TargetSpace(hilbert::Hilbert{<:Spin}, quantumnumbers::Union{Sz, Tuple{Sz, Vararg{Sz}}}, partition::NTuple{N, AbstractVector{Int}}=defaultpartition(length(hilbert))) where N
    quantumnumbers = wrapper(quantumnumbers)
    @assert all(quantumnumber->!isnan(quantumnumber.Sz), quantumnumbers) "TargetSpace error: when Sz is not conserved, no quantum number should be used."
    spins = sorted_spins(hilbert)
    quantumnumberses, permutations, records = intermediate(spins, partition)
    partition = ntuple(i->SpinSlice(partition[i], quantumnumberses[i], permutations[i]), Val(N))
    return TargetSpace([SpinBases(quantumnumber, spins, partition, records[quantumnumber]) for quantumnumber in quantumnumbers])
end

"""
    matrix(op::Operator, braket::NTuple{2, SpinBases}, table, dtype=valtype(op)) -> SparseMatrixCSC{dtype, Int}

Get the CSC-formed sparse matrix representation of an operator.

Here, `table` specifies the order of the operator ids.
"""
function matrix(op::Operator, braket::NTuple{2, SpinBases}, table, dtype=valtype(op))
    bra, ket = braket[1], braket[2]
    @assert match(bra, ket) "matrix error: mismatched bra and ket."
    @assert all(index->ket.spins[table[index]]==totalspin(index), id(op)) "matrix error: mismatched spin bases and operator."
    ms = [sparse(one(dtype)*I, Int(2*spin+1), Int(2*spin+1)) for spin in ket.spins]
    for (i, index) in enumerate(id(op))
        ms[table[index]] = i==1 ? op.value*sparse(matrix(index, dtype)) : sparse(matrix(index, dtype))
    end
    if isnan(AbelianNumber(bra).Sz)
        intermediate = eltype(ms)[]
        for slice in ket.partition
            for (i, position) in enumerate(slice.positions)
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
    else
        intermediate = map(spinslice->blocks(ms[spinslice.positions], spinslice.quantumnumbers, spinslice.permutation), ket.partition)
        result = SparseMatrixCOO(Int[], Int[], dtype[], length(bra), length(ket))
        for (row_keys, row_slice) in pairs(bra.record)
            for (col_keys, col_slice) in pairs(ket.record)
                kron!(result, map((row_key, col_key, m)->m[(row_key, col_key)], row_keys, col_keys, intermediate); origin=(row_slice.start, col_slice.start))
            end
        end
        return SparseMatrixCSC(result)
    end
end
function blocks(ms::Vector{<:SparseMatrixCSC}, quantumnumbers::AbelianNumbers{Sz}, permutation::Vector{Int})
    @assert quantumnumbers.form=='C'
    result = Dict{Tuple{Sz, Sz}, SparseMatrixCOO{eltype(eltype(ms)), Int}}()
    m = permute!(reduce(kron, ms), permutation, permutation)
    rows, vals = rowvals(m), nonzeros(m)
    for j = 1:length(quantumnumbers)
        temp = [(Int[], Int[], eltype(eltype(ms))[]) for i=1:length(quantumnumbers)]
        for (k, col) in enumerate(range(quantumnumbers, j))
            pos = 1
            for index in nzrange(m, col)
                row = rows[index]
                val = vals[index]
                pos = findindex(row, quantumnumbers, pos)
                push!(temp[pos][1], row-cumsum(quantumnumbers, pos-1))
                push!(temp[pos][2], k)
                push!(temp[pos][3], val)
            end
        end
        for (i, (is, js, vs)) in enumerate(temp)
            result[(quantumnumbers[i], quantumnumbers[j])] = SparseMatrixCOO(is, js, vs, count(quantumnumbers, i), count(quantumnumbers, j))
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

end # module
