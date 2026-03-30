module ExactDiagonalization

# Prerequisites
# Band Lanczos
include("BandLanczos.jl")

# Body of ExactDiagonalization
using Base: @propagate_inbounds
using Base.Iterators: product
using DataStructures: OrderedDict
using HalfIntegers: HalfInt
using KrylovKit: Block, eigsolve, expand!, initialize, normres, rayleighquotient
using LinearAlgebra: I, Hermitian, dot, norm
using LuxurySparse: SparseMatrixCOO
using Printf: @printf
using QuantumLattices: eager, efficientoperations, plain, azimuth, bonds, decompose, expand, idtype, indextype, internalindextype, iscreation, nneighbor, polar, reparameter, reset!, shape, statistics, str, totalspin
using QuantumLattices: 𝕊, AbstractLattice, Action, Algorithm, Assignment, Boundary, BrillouinZone, CategorizedGenerator, Combinations, CompositeDict, CompositeIndex, Data, DuplicatePermutations, Fock, FockIndex, Frontend, Generator, Hilbert, Index, Internal, InternalIndex, LinearTransformation, Matrixization, Neighbors, OneAtLeast, OneOrMore, Operator, OperatorIndex, OperatorIndexToTuple, OperatorPack, OperatorProd, Operators, OperatorSet, OperatorSum, QuantumOperator, ReciprocalSpace, ReciprocalZone, Spin, SpinIndex, Table, Term, VectorSpace, VectorSpaceDirectProducted, VectorSpaceDirectSummed, VectorSpaceEnumerative
using Random: seed!
using SparseArrays: SparseMatrixCSC, nnz, nonzeros, nzrange, rowvals, sparse, spzeros
using TimerOutputs: TimerOutput, @timeit
using .BandLanczos: BandLanczosFactorization, BandLanczosIterator

import LinearAlgebra: eigen
import QuantumLattices: Metric, Parameters, VectorSpaceStyle, ⊗, ⊕, add!, decompose, dimension, getcontent, id, kind, matrix, options, parameternames, partition, period, periods, rank, run!, scalartype, update!, value

## Quantum numbers
include("QuantumNumbers.jl")
export Abelian, AbelianQuantumNumber, AbelianQuantumNumberProd, AbelianGradedSpace, AbelianGradedSpaceProd, AbelianGradedSpaceSum, Graded, RepresentationSpace, SimpleAbelianQuantumNumber
export ⊕, ⊗, ⊠, ℕ, 𝕊ᶻ, 𝕌₁, ℤ, ℤ₁, fℤ₂, sℤ₂, decompose, dimension, findindex, period, periods, rank, regularize, regularize!, value

## Core
include("Core.jl")
export edtimer, eigen, id, kind, matrix, prepare!, productable, release!, scalartype, sumable, update!
export ED, EDKind, EDMatrix, EDMatrixization, Sector

## BinaryBases
include("BinaryBases.jl")
export BinaryBases, BinaryBasis

## AbelianBases
include("AbelianBases.jl")
export AbelianBases
export partition

## GreenFunctions
include("GreenFunctions.jl")
export GreenFunction, RetardedGreenFunction, set!
export GreenFunctionMethod, BandLanczosMethod, ExactMethod

## Assignments
include("Assignments.jl")
export EDEigen, EDEigenData
export GroundStateExpectation, GroundStateExpectationData
export SpinCoherentState, SpinCoherentStateProjection, SpinCoherentStateProjectionData
export StaticTwoPointCorrelator, StaticTwoPointCorrelatorData

end # module