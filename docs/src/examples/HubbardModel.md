```@meta
CurrentModule = ExactDiagonalization
```

# Fermi Hubbard Model on square lattice

## Ground state energy

The following codes could compute the ground state energy of the Fermi Hubbard model on square lattice.

```@example hubbard
using QuantumLattices
using ExactDiagonalization
using LinearAlgebra: eigen

# define the unitcell of the square lattice
unitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])

# define a finite 3×4 cluster of the square lattice with open boundary condition
lattice = Lattice(unitcell, (3, 4))

# define the Hilbert space (single-orbital spin-1/2 complex fermion)
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(lattice))

# define the quantum number of the sub-Hilbert space in which the computation to be carried out
# here the particle number is set to be `length(lattice)` and Sz is set to be 0
quantumnumber = SpinfulParticle(length(lattice), 0)

# define the terms, i.e. the nearest-neighbor hopping and the Hubbard interaction
t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 8.0)

# define the exact diagonalization algorithm for the Fermi Hubbard model
ed = ED(lattice, hilbert, (t, U), quantumnumber)

# find the ground state and its energy
eigensystem = eigen(matrix(ed); nev=1)

# Ground state energy should be -4.913259209075605
print(eigensystem.values)
```
