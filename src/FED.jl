module FED

using Printf: @printf, @sprintf
using SparseArrays: spzeros
using QuantumLattices: AbstractFockOperator, Operators, FOperator, CREATION, rank
using QuantumLattices.Prerequisites: Float
using QuantumLattices.Mathematics.VectorSpaces: VectorSpace

import SparseArrays: SparseMatrixCSC
import QuantumLattices: kind, id, dimension
import QuantumLattices.Mathematics.AlgebraOverFields: idtype

export FockBasis, GBasis, PBasis, SBasis, SPBasis

"""
    FockBasis{I<:Unsigned} <: VectorSpace{I}

Abatract type for all kinds of Fock basis.
"""
abstract type FockBasis{I<:Unsigned} <: VectorSpace{I} end
@inline kind(bs::FockBasis) = kind(typeof(bs))
@inline idtype(bs::FockBasis) = idtype(typeof(bs))
function Base.show(io::IO, bs::FockBasis)
    @printf io "%s:\n" repr(bs)
    for i = 1:dimension(bs)
        @printf io "  %s\n" string(bs[i], base=2)
    end
end
@inline Base.repr(bs::FockBasis) = @sprintf "%s(%s)" nameof(typeof(bs)) join(id(bs), ",")
@inline Base.issorted(bs::FockBasis) = true

"""
    GBasis{I}(nstate::Int) where I<:Unsigned

Generic fermionic/hard-core-bosonic basis, which uses no conserved quantities.
"""
struct GBasis{I<:Unsigned} <: FockBasis{I}
    nstate::Int
end
@inline kind(::Type{<:GBasis}) = :g
@inline idtype(::Type{<:GBasis}) = Int
@inline id(bs::GBasis) = bs.nstate
@inline dimension(bs::GBasis) = 2^bs.nstate
@inline Base.getindex(bs::GBasis, i::Int) = i-1
@inline Base.searchsortedfirst(bs::GBasis, i::Int) = i+1

"""
"""
struct PBasis{I<:Unsigned} <: FockBasis{I}
    nstate::Int
    nparticle::Int
    table::Vector{I}
end
@inline kind(::Type{<:PBasis}) = :p
@inline idtype(::Type{<:PBasis}) = Tuple{Int, Int}
@inline id(bs::PBasis) = (bs.nstate, bs.nparticle)
@inline dimension(bs::PBasis) = length(bs.table)
@inline Base.getindex(bs::GBasis, i::Int) = bs.table[i]

"""
"""
struct SBasis{I<:Unsigned} <: FockBasis{I}
    nstate::Int
    spinz::Rational{Int}
    table::Vector{I}
end
@inline kind(::Type{<:SBasis}) = :s
@inline idtype(::Type{<:SBasis}) = Tuple{Int,Rational{Int}}
@inline id(bs::SBasis) = (bs.nstate, bs.spinz)
@inline dimension(bs::SBasis) = length(bs.table)
@inline Base.getindex(bs::SBasis, i::Int) = bs.table[i]

"""
"""
struct SPBasis{I<:Unsigned} <: FockBasis{I}
    nstate::Int
    nparticle::Int
    spinz::Rational{Int}
    table::Vector{I}
end
@inline kind(::Type{<:SPBasis}) = :sp
@inline idtype(::Type{<:SPBasis}) = Tuple{Int, Int, Rational{Int}}
@inline id(bs::SPBasis) = (bs.nstate, bs.nparticle, bs.spinz)
@inline dimension(bs::SPBasis) = length(bs.table)
@inline Base.getindex(bs::SPBasis, i::Int) = bs.table[i]

"""
    SparseMatrixCSC(operator::AbstractFockOperator, basis::FockBasis, table) -> SparseMatrixCSC{valtype(operator), Int}
    SparseMatrixCSC(operator::AbstractFockOperator, basis₁::FockBasis, basis₂::FockBasis, table) -> SparseMatrixCSC{valtype(operator), Int}
    SparseMatrixCSC(opts::Operators, basis::FockBasis, table, dtype::Type{<:Number}=Complex{Float}) -> SparseMatrixCSC{dtype, Int}
    SparseMatrixCSC(opts::Operators, basis₁::FockBasis, basis₂::FockBasis, table, dtype::Type{<:Number}=Complex{Float}) -> SparseMatrixCSC{dtype, Int}

Get the CSC-formed sparse matrix representation of
1) an even-ranked operator,
2) an odd-ranked operator,
3) a set of even-ranked operators,
4) a set of odd-ranked operators.
"""
function SparseMatrixCSC(operator::AbstractFockOperator, basis::FockBasis, table)
    @assert rank(operator)%2==0 "SparseMatrixCSC error: two bases are needed for odd ranked Fock operators."
    ndata, data, indices, indptr = 1, zeros(valtype(operator), dimension(basis)), zeros(Int, dimension(basis)), zeros(Int, dimension(basis)+1)
    eye, temp, flag = one(basis|>eltype), zeros(basis|>eltype, rank(operator)+1), true
    seqs = NTuple{rank(operator), Int}(table[id(operator)[i]]-1 for i = 1:rank(operator))
    nambus = NTuple{rank(operator), Bool}(oid.index.nambu==CREATION for oid in id(operator))
    for i = 1:dimension(basis)
        flag = true
        indptr[i] = ndata
        temp[1] = basis[i]
        for j = 1:rank(operator)
            (temp[j]&eye<<seqs[j]>0)==nambus[j] && (flag = false; break)
            temp[j+1] = nambus[j] ? temp[j]|eye<<seqs[j] : temp[j]&~(eye<<seqs[j])
        end
        if flag
            nsign = 0
            isa(operator, FOperator) && for j = 1:rank(operator)
                for k = 0:seqs[j]-1
                    temp[j]&eye<<k>0 && (nsign += 1)
                end
            end
            indices[ndata] = searchsortedfirst(basis, temp[end])
            data[ndata] = operator.value*(-1)^nsign
            ndata += 1
        end
    end
    indptr[end] = ndata+1
    return SparseMatrixCSC(dimension(basis), dimension(basis), indptr, indices[1:ndata], data[1:ndata])
end
function SparseMatrixCSC(operator::AbstractFockOperator, basis₁::FockBasis, basis₂::FockBasis, table)
    @assert rank(operator)%2==1 "SparseMatrixCSC error: only one basis is needed for even ranked Fock operators."
    ndata, data, indices, indptr = 1, zeros(valtype(operator), dimension(basis₁)), zeros(Int, dimension(basis₁)), zeros(Int, dimension(basis₁)+1)
    eye, temp, flag = one(basis₁|>eltype), zeros(basis₁|>eltype, rank(operator)+1), true
    seqs = NTuple{rank(operator), Int}(table[id(operator)id[i]] for i = 1:rank(operator))
    nambus = NTuple{rank(operator), Bool}(oid.index.nambu==CREATION for oid in id(operator))
    for i = 1:dimension(basis₁)
        flag = true
        indptr[i] = ndata
        temp[1] = basis₁[i]
        for j = 1:rank(operator)
            (temp[j]&eye<<seqs[j]>0)==nambus[j] && (flag = false; break)
            temp[j+1] = nambus[j] ? temp[j]|eye<<seqs[j] : temp[j]&~(eye<<seqs[j])
        end
        if flag
            nsign = 0
            isa(operator,FOperator) && for j = 1:rank(operator)
                for k = 0:seqs[j]-1
                    temp[j]&eye<<k>0 && (nsign += 1)
                end
            end
            indices[ndata] = searchsortedfirst(basis₂, temp[end])
            data[ndata] = operator.value*(-1)^nsign
            ndata += 1
        end
    end
    indptr[end] = ndata+1
    return SparseMatrixCSC(dimension(basis₂), dimension(basis₁), indptr, indices[1:ndata], data[1:ndata])
end
function SparseMatrixCSC(opts::Operators, basis::FockBasis, table, dtype::Type{<:Number}=Complex{Float})
    result = spzeros(dtype, dimension(basis), dimension(basis))
    for opt in values(opts)
        result += SparseMatrixCSC(opt, basis, table)
    end
    return result
end
function SparseMatrixCSC(opts::Operators, basis₁::FockBasis, basis₂::FockBasis, table, dtype::Type{<:Number}=Complex{Float})
    result = spzeros(dtype, dimension(basis₂), dimension(basis₁))
    for opt in values(opts)
        result += matrix(opt, basis₁, basis₂, table)
    end
    return result
end


end  # module
