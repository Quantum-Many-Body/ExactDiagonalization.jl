# ExactDiagonalization

[![CI](https://github.com/Quantum-Many-Body/ExactDiagonalization.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Quantum-Many-Body/ExactDiagonalization.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/Quantum-Many-Body/ExactDiagonalization.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Quantum-Many-Body/ExactDiagonalization.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://quantum-many-body.github.io/ExactDiagonalization.jl/dev/)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://quantum-many-body.github.io/ExactDiagonalization.jl/stable/)
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)
[![LICENSE](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

*Julia package for the exact diagonalization method in condensed matter physics.*

## Introduction

Based on the **symbolic operator representation** of a quantum lattice system in condensed matter physics that is generated by the package [`QuantumLattices`](https://github.com/Quantum-Many-Body/QuantumLattices.jl), exact diagonalization method is implemented for fermionic, hard-core-bosonic and spin systems.

## Installation

In Julia **v1.8+**, please type `]` in the REPL to use the package mode, then type this command:

```julia
pkg> add ExactDiagonalization
```

## Getting Started

[Examples of exact diagonalization method for quantum lattice system](@ref examples)

## Note

Due to the fast development of this package, releases with different minor version numbers are **not** guaranteed to be compatible with previous ones **before** the release of v1.0.0. Comments are welcomed in the issues.

## Contact
waltergu1989@gmail.com

## Python counterpart
[HamiltonianPy](https://github.com/waltergu/HamiltonianPy): in fact, the authors of this Julia package worked on the python package at first and only turned to Julia later.