```@meta
CurrentModule = ExactDiagonalization
```

# Antiferromagnetic Heisenberg Model on square lattice

## Ground state energy

The following codes could compute the ground state energy of the antiferromagnetic Heisenberg model on square lattice.

```@example heisenberg
using QuantumLattices
using ExactDiagonalization
using LinearAlgebra: eigen

# define the unitcell of the square lattice
unitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])

# define a finite 4×4 cluster of the square lattice with open boundary condition
lattice = Lattice(unitcell, (4, 4))

# define the Hilbert space (spin-1/2)
hilbert = Hilbert(Spin{1//2}(), length(lattice))

# define the quantum number of the sub-Hilbert space in which the computation to be carried out
# for the ground state, Sz=0
quantumnumber = Sz(0)

# define the antiferromagnetic Heisenberg term on the nearest neighbor
J = Heisenberg(:J, 1.0, 1)

# define the exact diagonalization algorithm for the antiferromagnetic Heisenberg model
ed = ED(lattice, hilbert, J, quantumnumber)

# find the ground state and its energy
eigensystem = eigen(ed; nev=1)

# Ground state energy should be -9.189207065192935
print(eigensystem.values)
```
