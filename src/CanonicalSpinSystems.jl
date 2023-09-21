module CanonicalSpinSystems

using QuantumLattices: Hilbert, Metric, OperatorUnitToTuple, Spin, SpinTerm
using ..EDCore: EDKind

"""
    EDKind(::Type{<:SpinTerm})

The kind of the exact diagonalization method applied to a canonical quantum spin lattice system.
"""
@inline EDKind(::Type{<:SpinTerm}) = EDKind(:SED)

"""
    Metric(::EDKind{:SED}, ::Hilbert{<:Spin}) -> OperatorUnitToTuple

Get the index-to-tuple metric for a canonical quantum spin lattice system.
"""
@inline @generated Metric(::EDKind{:SED}, ::Hilbert{<:Spin}) = OperatorUnitToTuple(:site)

end # module
