module CanonicalSpinSystems

using LinearAlgebra: I
using Printf: @printf
using QuantumLattices: totalspin
using QuantumLattices: AbelianNumber, AbelianNumbers, Hilbert, Metric, Operator, OperatorUnitToTuple, Spin, Sz
using SparseArrays: sparse
using ..EDCore: EDKind, Sector, TargetSpace

import QuantumLattices: id, matrix
import ..EDCore: sumable

export SpinBases

"""
    SpinBases <: Sector

A set of spin bases.
"""
struct SpinBases <: Sector
    quantumnumber::Sz
    spins::Vector{Rational{Int}}
    partition::Vector{Vector{Int}}
    permutations::Vector{Vector{Int}}
    permutation::Vector{Int}
    slice::Vector{Int}
    function SpinBases(quantumnumber::Sz, spins::Vector{Rational{Int}}, partition::Vector{Vector{Int}}, permutations::Vector{Vector{Int}}, permutation::Vector{Int}, slice::Vector{Int})
        @assert Set(vcat(partition...))==Set(1:length(spins)) "SpinBases error: incorrect partition."
        new(quantumnumber, spins, partition, permutations, permutation, slice)
    end
end
@inline id(bs::SpinBases) = (bs.quantumnumber, bs.spins, bs.partition)
@inline Base.length(bs::SpinBases) = length(bs.slice)
function Base.show(io::IO, bs::SpinBases)
    tostr(spin::Rational{Int}, order::Int) = string(spin.den==1 ? string(spin.num) : string(spin.num, "/", spin.den), join('₀'+d for d in reverse(digits(order))))
    @printf io "%s" "{"
    for (i, group) in enumerate(bs.partition)
        @printf io "(%s)" join([tostr(bs.spins[order], order) for order in group], "⊗")
        i<length(bs.partition) && @printf io "%s" " ⊗ "
    end
    @printf io ": %s}" bs.quantumnumber
end

"""
    SpinBases(spins::Vector{<:Real})
    SpinBases(spins::Vector{<:Real}, partition::Vector{<:AbstractVector{Int}})
    SpinBases(quantumnumber::Sz, spins::Vector{<:Real})
    SpinBases(quantumnumber::Sz, spins::Vector{<:Real}, partition::Vector{<:AbstractVector{Int}})

Construct a set of spin bases.
"""
@inline SpinBases(spins::Vector{<:Real}) = SpinBases(Sz(NaN), spins)
@inline SpinBases(spins::Vector{<:Real}, partition::Vector{<:AbstractVector{Int}}) = SpinBases(Sz(NaN), spins, partition)
@inline SpinBases(quantumnumber::Sz, spins::Vector{<:Real}) = SpinBases(quantumnumber, spins, defaultpartition(length(spins)))
function SpinBases(quantumnumber::Sz, spins::Vector{<:Real}, partition::Vector{<:AbstractVector{Int}})
    spins = collect(Rational{Int}, spins)
    @assert all(spin->spin.den∈(1, 2), spins) "SpinBases error: incorrect input spins."
    partition = convert(Vector{Vector{Int}}, partition)
    if isnan(quantumnumber.Sz)
        permutations, permutation, slice = Vector{Int}[], Int[], Int[]
    else
        total, permutation, permutations = intermediate(spins, partition)
        slice = findall(quantumnumber, total, :expansion)
    end
    return SpinBases(quantumnumber, spins, partition, permutations, permutation, slice)
end
@inline defaultpartition(n::Int) = n==1 ? [[1]] : (cut=n÷2; [collect(1:cut), collect(cut+1:n)])
function intermediate(spins, partition)
    qnses, permutations = AbelianNumbers{Sz}[], Vector{Int}[]
    for group in partition
        qns, permutation = sort(kron([spinzs(spins[order]) for order in group]...))
        push!(qnses, qns)
        push!(permutations, permutation)
    end
    return sort(kron(qnses...))..., permutations
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
    Sector(hilbert::Hilbert{<:Spin}; kwargs...)
    Sector(hilbert::Hilbert{<:Spin}, quantumnumber::Sz; kwargs...)

Construct the spin bases of a Hilbert space with the specified quantum number.
"""
@inline Sector(hilbert::Hilbert{<:Spin}; kwargs...) = Sector(hilbert, Sz(NaN); kwargs...)
function Sector(hilbert::Hilbert{<:Spin}, quantumnumber::Sz; kwargs...)
    permutation = sortperm(collect(keys(hilbert)))
    spins = permute!([convert(Rational{Int}, totalspin(spin)) for spin in values(hilbert)], permutation)
    return SpinBases(quantumnumber, spins)
end

"""
    TargetSpace(hilbert::Hilbert{<:Spin}, quantumnumber::Sz, quantumnumbers::Sz...; kwargs...) -> TargetSpace

Construct a target space from the total Hilbert space and the associated quantum numbers.
"""
function TargetSpace(hilbert::Hilbert{<:Spin}, quantumnumber::Sz, quantumnumbers::Sz...; kwargs...)
    quantumnumbers = (quantumnumber, quantumnumbers...)
    @assert all(quantumnumber->!isnan(quantumnumber.Sz), quantumnumbers) "TargetSpace error: when Sz is not conserved, it should be omitted."
    spins = permute!([convert(Rational{Int}, totalspin(spin)) for spin in values(hilbert)], sortperm(collect(keys(hilbert))))
    partition = defaultpartition(length(spins))
    total, permutation, permutations = intermediate(spins, partition)
    return TargetSpace([SpinBases(quantumnumber, spins, partition, permutations, permutation, findall(quantumnumber, total, :expansion)) for quantumnumber in quantumnumbers])
end

"""
    matrix(op::Operator, braket::NTuple{2, SpinBases}, table; dtype=valtype(op)) -> SparseMatrixCSC{dtype, Int}

Get the CSC-formed sparse matrix representation of an operator.

Here, `table` specifies the order of the operator ids.
"""
function matrix(op::Operator, braket::NTuple{2, SpinBases}, table; dtype=valtype(op))
    bra, ket = braket[1], braket[2]
    @assert bra.spins==ket.spins && bra.partition==ket.partition "matrix error: mismatched bra and ket."
    @assert all(index->ket.spins[table[index]]==totalspin(index), id(op)) "matrix error: mismatched spin bases and operator."
    ms = [sparse(one(dtype)*I, Int(2*spin+1), Int(2*spin+1)) for spin in ket.spins]
    for index in id(op)
        ms[table[index]] = sparse(matrix(index, dtype))
    end
    intermediate = [permute!(reduce(kron, ms[group]), permutation, permutation) for (group, permutation) in zip(ket.partition, ket.permutations)]
    # return reduce(kron, intermediate)
    return reduce(kron, intermediate)[bra.permutation, ket.permutation]
end

end # module
