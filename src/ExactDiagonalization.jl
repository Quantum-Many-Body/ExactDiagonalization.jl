module ExactDiagonalization

# Prerequisites
# Band Lanczos
include("BandLanczos.jl")

# Quantum numbers
include("QuantumNumbers.jl")
export Abelian, AbelianQuantumNumber, AbelianQuantumNumberProd, AbelianGradedSpace, AbelianGradedSpaceProd, AbelianGradedSpaceSum, Graded, RepresentationSpace, SimpleAbelianQuantumNumber
export ⊕, ⊗, ⊠, ℕ, 𝕊ᶻ, 𝕌₁, ℤ, ℤ₁, fℤ₂, sℤ₂, decompose, dimension, findindex, period, periods, rank, regularize, regularize!, value

# Core
include("Core.jl")
export edtimer, eigen, prepare!, productable, release!, sumable
export AbelianBases, BinaryBases, BinaryBasis, ED, EDKind, EDMatrix, EDMatrixization, Sector, SpinCoherentState
export EDEigen, EDEigenData, GroundStateExpectation, GroundStateExpectationData, SpinCoherentStateProjection, SpinCoherentStateProjectionData, StaticTwoPointCorrelator, StaticTwoPointCorrelatorData

end # module