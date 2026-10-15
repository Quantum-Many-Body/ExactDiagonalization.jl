module ExactDiagonalization

# Prerequisites
# Band Lanczos
include("BandLanczos.jl")

# Body of ExactDiagonalization
using Base: @propagate_inbounds
using Base.Iterators: product
using DataStructures: OrderedDict
using HalfIntegers: HalfInt
using KrylovKit: Block, ModifiedGramSchmidt, ModifiedGramSchmidt2, Orthogonalizer, eigsolve, expand!, initialize, normres, rayleighquotient
using LinearAlgebra: I, Hermitian, norm, normalize!
using LuxurySparse: SparseMatrixCOO
using Mmap: mmap
using Printf: @printf
using QuantumLattices: efficientoperations, plain, azimuth, bonds, decompose, expand, idtype, indextype, internalindextype, iscreation, nneighbor, polar, reparameter, reset!, shape, statistics, str, totalspin
using QuantumLattices: 𝕊, AbstractLattice, Algorithm, Assignment, Boundary, BrillouinZone, CategorizedGenerator, Combinations, CompositeDict, CompositeIndex, Data, DuplicatePermutations, Fock, FockIndex, Generator, Hilbert, Index, Internal, InternalIndex, LatticeModel, LinearTransformation, Matrixization, Neighbors, OneAtLeast, OneOrMore, Operator, OperatorGenerator, OperatorIndex, OperatorIndexToTuple, OperatorPack, OperatorProd, Operators, OperatorSet, OperatorSum, Pairing, QuantumOperator, ReciprocalSpace, ReciprocalZone, Spin, SpinIndex, Table, Term, VectorSpace, VectorSpaceDirectProducted, VectorSpaceDirectSummed, VectorSpaceEnumerative, @delegate
using Random: randn, seed!
using Serialization: serialize
using SparseArrays: AbstractSparseMatrix, SparseMatrixCSC, nonzeros, nzrange, rowvals, spzeros
using TimerOutputs: TimerOutput, @timeit
using .BandLanczos: BandLanczosFactorization, BandLanczosIterator, basis

import LinearAlgebra: dot, eigen, mul!
import QuantumLattices: Metric, Parameters, VectorSpaceStyle, ⊗, ⊕, add!, contenttoconfig, decompose, dependencytypes, dimension, getcontent, id, kind, matrix, options, parameternames, partition, period, periods, rank, run!, scalartype, showasleaf, showcontent, update!, value
import SparseArrays: nnz, sparse

## Quantum numbers
include("QuantumNumbers.jl")
export Abelian, AbelianQuantumNumber, AbelianQuantumNumberProd, AbelianGradedSpace, AbelianGradedSpaceProd, AbelianGradedSpaceSum, Graded, RepresentationSpace, SimpleAbelianQuantumNumber
export ⊕, ⊗, ⊠, ℕ, 𝕊ᶻ, 𝕌₁, ℤ, ℤ₁, fℤ₂, sℤ₂, decompose, dimension, findindex, period, periods, rank, regularize, regularize!, value

## DiskCache
include("DiskCache.jl")
export DiskCache, DiskCSCMatrix, diskcache, diskcache!

## Core
include("Core.jl")
export edtimer, eigen, id, kind, matrix, normalize, prepare!, productable, release!, scalartype, sparse, sumable, update!
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
export GreenFunction, GreenFunctionKind, GreaterOrLesserKind
export GreenFunctionMethod, BandLanczosMethod, ExactMethod, ThermalExactMethod, ThermalSeedBandLanczosMethod

## Assignments
include("Assignments.jl")
export EDEigen, EDEigenData
export GroundStateExpectation, GroundStateExpectationData
export SpinCoherentState, SpinCoherentStateProjection, SpinCoherentStateProjectionData
export StaticTwoPointCorrelator, StaticTwoPointCorrelatorData

end # module