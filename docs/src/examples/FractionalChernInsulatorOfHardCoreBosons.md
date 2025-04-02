```@meta
CurrentModule = ExactDiagonalization
```

# Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice

## Many-body energy levels as a function of twisted boundary conditions

The following codes could compute the many-body energy levels as a function of twisted boundary conditions of hard-core bosons on honeycomb lattice with a nearly-flat Chern band.

```@example fci
using QuantumLattices
using ExactDiagonalization
using LinearAlgebra: eigen
using Plots: plot

# unit cell of the honeycomb lattice
unitcell = Lattice([0.0, 0.0], [0.0, √3/3]; vectors=[[1.0, 0.0], [0.5, √3/2]])

# 4×3 cluster
lattice = Lattice(unitcell, (4, 3), ('P', 'P'))

# twisted boundary condition
boundary = Boundary{(:θ₁, :θ₂)}([0.0, 0.0], lattice.vectors)

# Hilbert space of hard-core bosons
hilbert = Hilbert(Fock{:b}(1, 1), length(lattice))

# Parameters of the model, with hoppings up to 3rd nearest neighbor and interactions up to 2nd
# the 2nd nearest neighbor hopping is complex, similar to that of the Haldane model
# with such parameters, the model host a nearly flat band with ±1 Chern number
parameters = (
    t=Complex(-1.0), t′=Complex(-0.6), t′′=Complex(0.58), φ=0.4, V₁=1.0, V₂=0.4, θ₁=0.0, θ₂=0.0
)
map(parameters::NamedTuple) = (
    t₁=parameters.t,
    t₂=parameters.t′*cos(parameters.φ*pi),
    λ₂=parameters.t′*sin(parameters.φ*pi),
    t₃=parameters.t′′,
    V₁=parameters.V₁,
    V₂=parameters.V₂,
    θ₁=parameters.θ₁,
    θ₂=parameters.θ₂
)

# terms
t₁ = Hopping(:t₁, map(parameters).t₁, 1; ismodulatable=false)
t₂ = Hopping(:t₂, map(parameters).t₂, 2; ismodulatable=false)
λ₂ = Hopping(:λ₂, map(parameters).λ₂, 2;
    amplitude=bond::Bond->1im*cos(3*azimuth(rcoordinate(bond)))*(-1)^(bond[1].site%2),
    ismodulatable=false
)
t₃ = Hopping(:t₃, map(parameters).t₃, 3; ismodulatable=false)
V₁ = Coulomb(:V₁, map(parameters).V₁, 1)
V₂ = Coulomb(:V₂, map(parameters).V₂, 2)

# 1/4 filling of the model, i.e., half-filling of the lower flat Chern band
# In such case, the ground state of the model is a bosonic fractional Chern insulator
# see Y.-F. Wang, et al. PRL 107, 146803 (2011)
quantumnumber = ℕ(length(lattice)÷4)

# construct the algorithm
fci = Algorithm(
    :FCI,
    ED(lattice, hilbert, (t₁, t₂, λ₂, t₃, V₁, V₂), quantumnumber, boundary),
    parameters,
    map
)

# define the boundary angles and number of energy levels to be computed
nθ, nev = 5, 3
θs = range(0, 2, nθ)
data = zeros(nθ, nev)

# compute the energy levels with different twist boundary angles
for (i, θ) in enumerate(θs)
    update!(fci; θ₁=θ)
    data[i, :] = eigen(fci; nev=nev).values
end

# plot the result
plot(θs, data; legend=false)
```
