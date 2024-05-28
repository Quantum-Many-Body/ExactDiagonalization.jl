module CanonicalSpinSystems

using Base.Iterators: product
using LinearAlgebra: I, dot, norm
using LuxurySparse: SparseMatrixCOO
using Printf: @printf
using QuantumLattices: findindex, totalspin
using QuantumLattices: AbelianNumber, AbelianNumbers, Hilbert, Metric, Operator, OperatorUnitToTuple, Spin, Sz
using QuantumLattices: AbstractLattice, Table, Index, polar, azimuth
using SparseArrays: SparseMatrixCSC, nnz, nonzeros, nzrange, rowvals, sparse
using ..EDCore: EDKind, Sector, TargetSpace, wrapper

import QuantumLattices: id, matrix
import ..EDCore: sumable

export SpinBases
export xyz2ang, spincoherentstates, structure_factor, Pspincoherentstates

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

function xyz2ang(spins::Dict{Int, Vector{T}}) where {T<:Real}
    out = Matrix{Float64}(undef, 2, length(spins))
    for (i, k) in spins
        @assert length(k)==3 "xyz2ang error: incomplete spin components."
        out[1, i] = polar(k)
        out[2, i] = azimuth(k)
    end
    return out
end
function xyz2ang(spins::Matrix{T}) where {T<:Real}
    @assert size(spins, 1)==3 "xyz2ang error: incomplete spin components."
    out = Matrix{Float64}(undef, 2, size(spins, 2))
    for i = size(spins, 2)
        k = spins[:, i]
        out[1, i] = polar(k)
        out[2, i] = azimuth(k)
    end
    return out
end

"""
    spincoherentstates(structure::Matrix{Float64}) -> Matrix{Float64}

Get the spin coherent states from the input spin structures specified by the polar and azimuth angles.
"""
function spincoherentstates(structure::Matrix{Float64})
    @assert size(structure, 1)==2 "spincoherentstates error: spin structures must be specified by the polar and azimuth angles of a spin orientation."
    out = [[exp(im/2*structure[2, i])*sin(structure[1, i]/2), exp(-im/2*structure[2, i])*cos(structure[1, i]/2)] for i=1:size(structure, 2)]
    return kron(out...)
end

"""
    structure_factor(lattice::AbstractLattice, bs::SpinBases, hilbert::Hilbert, scs::AbstractVector{T}, k::Vector{Float64}) where {T<:Number} -> [SxSx(k), SySy(k), SzSz(k)]
    structure_factor(lattice::AbstractLattice, bs::SpinBases, hilbert::Hilbert, scs::AbstractVector{T}; Nk::Int=60) where {T<:Number} -> Matrix(3, Nk, Nk)

Get structure_factor of state "scs".
"""
function structure_factor(lattice::AbstractLattice, bs::SpinBases, hilbert::Hilbert, scs::AbstractVector{T}, k::Vector{Float64}) where {T<:Number}
    N = length(lattice)
    table = Table(hilbert, OperatorUnitToTuple(:site))
    base = (bs, bs)
    sq = zeros(ComplexF64, 3)
    for j=1:N, i=1:N
        phase = exp(im*dot(k, lattice[i]-lattice[j]))
        xx = Operator(1, Index(i, hilbert[i][1]), Index(j, hilbert[j][1]))
        yy = Operator(1, Index(i, hilbert[i][2]), Index(j, hilbert[j][2]))
        zz = Operator(1, Index(i, hilbert[i][3]), Index(j, hilbert[j][3]))
        mx = matrix(xx, base, table, ComplexF64)
        my = matrix(yy, base, table, ComplexF64)
        mz = matrix(zz, base, table, ComplexF64)
        sq[1] += real(dot(scs, mx, scs))*phase
        sq[2] += real(dot(scs, my, scs))*phase
        sq[3] += real(dot(scs, mz, scs))*phase
    end
    return real.(sq)/N
end
function structure_factor(lattice::AbstractLattice, bs::SpinBases, hilbert::Hilbert, scs::AbstractVector{T}; Nk::Int=60) where {T<:Number}
    N = length(lattice)
    table = Table(hilbert, OperatorUnitToTuple(:site))
    base = (bs, bs)
    ks = range(-2pi, 2pi, length=Nk+1)
    ss = Array{Float64}(undef, 3, N, N)
    for j=1:N, i=1:N
        xx = Operator(1, Index(i, hilbert[i][1]), Index(j, hilbert[j][1]))
        yy = Operator(1, Index(i, hilbert[i][2]), Index(j, hilbert[j][2]))
        zz = Operator(1, Index(i, hilbert[i][3]), Index(j, hilbert[j][3]))
        mx = matrix(xx, base, table, ComplexF64)
        ss[1, i, j] = real(dot(scs, mx, scs))
        my = matrix(yy, base, table, ComplexF64)
        ss[2, i, j] = real(dot(scs, my, scs))
        mz = matrix(zz, base, table, ComplexF64)
        ss[3, i, j] = real(dot(scs, mz, scs))
    end
    sq = zeros(ComplexF64, 3, Nk+1, Nk+1)
    for x=1:Nk+1, y=1:Nk+1
        ki = [ks[x], ks[y]]
        for j=1:N, i=1:N
            phase = exp(im*dot(ki, lattice[i]-lattice[j]))
            sq[1, x, y] += ss[1,i,j] * phase
            sq[2, x, y] += ss[2,i,j] * phase
            sq[3, x, y] += ss[3,i,j] * phase
        end
    end
    return ks, real.(sq)/N
end

"""
    Pspincoherentstates(scs::AbstractVector{T}, spins::Dict{Vector{Int}, Vector{Float64}}; N::Int=100) where {T<:Number}

Get square of the Projectors of state "scs" onto spincoherentstates.
"""
function Pspincoherentstates(scs::AbstractVector{T}, spins::Dict{Vector{Int}, Vector{Float64}}; N::Int=100) where {T<:Number}
    L = spins|>keys.|>length|>sum
    @assert sort(cat(keys(spins)...,dims=1))==1:L|>collect "Pspincoherentstates error: the lattices are not matching."
    s = range(0, pi, length=N)
    p = range(0, 2*pi, length=N)
    out = Matrix{Float64}(undef, N, N)
    for i=1:N, j=1:N
        ss = zeros(2, L)
        for (k, v) in spins
            ss[:, k] .= [s[i], p[j]] + v
        end
        scst = spincoherentstates(ss)
        out[j, i] = abs(dot(scst, scs))^2
    end
    return s, p, out
end

end # module
