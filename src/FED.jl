module FED

using Printf: @printf,@sprintf
using SparseArrays: SparseMatrixCSC,spzeros
using QuantumLattices: FockOperator,Operators,FOperator,MAJORANA,CREATION,rank
using QuantumLattices.Prerequisites: Float
using QuantumLattices.Mathematics.VectorSpaces: VectorSpace

import QuantumLattices: kind,id,dimension,matrix
import QuantumLattices.Mathematics.AlgebraOverFields: idtype

export FockBasis,GBasis

"""
    FockBasis{I<:Unsigned} <: VectorSpace{I}

Abatract type for all kinds of Fock basis.
"""
abstract type FockBasis{I<:Unsigned} <: VectorSpace{I} end
kind(bs::FockBasis)=bs|>typeof|>kind
idtype(bs::FockBasis)=bs|>typeof|>idtype
function Base.show(io::IO,bs::FockBasis)
    @printf io "%s:\n" repr(bs)
    for i=1:dimension(bs)
        @printf io "  %s\n" string(bs[i],base=2)
    end
end
Base.repr(bs::FockBasis)=@sprintf "%s(%s)" nameof(typeof(bs)) join(id(bs),",")

"""
    GBasis{I}(nstate::Int) where I<:Unsigned

Generic fermionic/hard-core-bosonic basis, which uses no conserved quantities.
"""
struct GBasis{I<:Unsigned} <: FockBasis{I}
    nstate::Int
end
dimension(bs::GBasis)=2^bs.nstate
Base.getindex(bs::GBasis,i::Int)=i-1
Base.searchsortedfirst(bs::GBasis,i::Int)=i+1
kind(::Type{<:GBasis})=:g
idtype(::Type{<:GBasis})=Int
id(bs::GBasis)=bs.nstate

"""
    matrix(operator::FockOperator,basis::FockBasis) -> SparseMatrixCSC{valtype(operator),Int}
    matrix(opts::Operators,basis::FockBasis,dtype::Type{<:Number}=Complex{Float}) -> SparseMatrixCSC{dtype,Int}

Get the CSC-formed sparse matrix representation of an operator / a set of operators.
"""
function matrix(operator::FockOperator,basis::FockBasis)
    @assert rank(operator)%2==0 "matrix error: two bases are needed for odd ranked Fock operators."
    ndata,data,indices,indptr=1,zeros(valtype(operator),dimension(basis)),zeros(Int,dimension(basis)),zeros(Int,dimension(basis)+1)
    eye,temp,flag=one(basis|>eltype),zeros(basis|>eltype,rank(operator)+1),true
    seqs=NTuple{rank(operator),Int}(seq-1 for seq in operator.id.seqs)
    nambus=NTuple{rank(operator),Bool}((oid.index.nambu==MAJORANA ? error("matrix error: majorana operator not supported.") : oid.index.nambu==CREATION) for oid in operator.id)
    for i=1:dimension(basis)
        flag=true
        indptr[i]=ndata
        temp[1]=basis[i]
        for j=1:rank(operator)
            (temp[j]&eye<<seqs[j]>0)==nambus[j] && (flag=false; break)
            temp[j+1]=nambus[j] ? temp[j]|eye<<seqs[j] : temp[j]&~(eye<<seqs[j])
        end
        if flag
            nsign=0
            isa(operator,FOperator) && for j=1:rank(operator) for k=0:seqs[j]-1 (temp[j]&eye<<k>0 && (nsign+=1)) end end
            indices[ndata]=searchsortedfirst(basis,temp[end])
            data[ndata]=(-1)^nsign*operator.value
            ndata+=1
        end
    end
    indptr[end]=ndata+1
    return SparseMatrixCSC(dimension(basis),dimension(basis),indptr,indices[1:ndata],data[1:ndata])
end
function matrix(opts::Operators,basis::FockBasis,dtype::Type{<:Number}=Complex{Float})
    result=spzeros(dtype,dimension(basis),dimension(basis))
    for opt in values(opts) result=result+matrix(opt,basis) end
    return result
end

end  # module
