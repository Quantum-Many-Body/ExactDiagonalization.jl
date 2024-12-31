var documenterSearchIndex = {"docs":
[{"location":"examples/FractionalChernInsulatorOfHardCoreBosons/","page":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","title":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","text":"CurrentModule = ExactDiagonalization","category":"page"},{"location":"examples/FractionalChernInsulatorOfHardCoreBosons/#Fractional-Chern-Insulator-of-Hard-core-Bosons-on-honeycomb-lattice","page":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","title":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","text":"","category":"section"},{"location":"examples/FractionalChernInsulatorOfHardCoreBosons/#Many-body-energy-levels-as-a-function-of-twisted-boundary-conditions","page":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","title":"Many-body energy levels as a function of twisted boundary conditions","text":"","category":"section"},{"location":"examples/FractionalChernInsulatorOfHardCoreBosons/","page":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","title":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","text":"The following codes could compute the many-body energy levels as a function of twisted boundary conditions of hard-core bosons on honeycomb lattice with a nearly-flat Chern band.","category":"page"},{"location":"examples/FractionalChernInsulatorOfHardCoreBosons/","page":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","title":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","text":"using QuantumLattices\nusing ExactDiagonalization\nusing LinearAlgebra: eigen\nusing Plots: plot\n\n# unit cell of the honeycomb lattice\nunitcell = Lattice([0.0, 0.0], [0.0, √3/3]; vectors=[[1.0, 0.0], [0.5, √3/2]])\n\n# 4×3 cluster\nlattice = Lattice(unitcell, (4, 3), ('P', 'P'))\n\n# twisted boundary condition\nboundary = Boundary{(:θ₁, :θ₂)}([0.0, 0.0], lattice.vectors)\n\n# Hilbert space of hard-core bosons\nhilbert = Hilbert(Fock{:b}(1, 1), length(lattice))\n\n# Parameters of the model, with hoppings up to 3rd nearest neighbor and interactions up to 2nd\n# the 2nd nearest neighbor hopping is complex, similar to that of the Haldane model\n# with such parameters, the model host a nearly flat band with ±1 Chern number\nparameters = (\n    t=Complex(-1.0), t′=Complex(-0.6), t′′=Complex(0.58), φ=0.4, V₁=1.0, V₂=0.4, θ₁=0.0, θ₂=0.0\n)\nmap(parameters::NamedTuple) = (\n    t₁=parameters.t,\n    t₂=parameters.t′*cos(parameters.φ*pi),\n    λ₂=parameters.t′*sin(parameters.φ*pi),\n    t₃=parameters.t′′,\n    V₁=parameters.V₁,\n    V₂=parameters.V₂,\n    θ₁=parameters.θ₁,\n    θ₂=parameters.θ₂\n)\n\n# terms\nt₁ = Hopping(:t₁, map(parameters).t₁, 1; ismodulatable=false)\nt₂ = Hopping(:t₂, map(parameters).t₂, 2; ismodulatable=false)\nλ₂ = Hopping(:λ₂, map(parameters).λ₂, 2;\n    amplitude=bond::Bond->1im*cos(3*azimuth(rcoordinate(bond)))*(-1)^(bond[1].site%2),\n    ismodulatable=false\n)\nt₃ = Hopping(:t₃, map(parameters).t₃, 3; ismodulatable=false)\nV₁ = Coulomb(:V₁, map(parameters).V₁, 1)\nV₂ = Coulomb(:V₂, map(parameters).V₂, 2)\n\n# 1/4 filling of the model, i.e., half-filling of the lower flat Chern band\n# In such case, the ground state of the model is a bosonic fractional Chern insulator\n# see Y.-F. Wang, et al. PRL 107, 146803 (2011)\nquantumnumber = ℕ(length(lattice)÷4)\n\n# construct the algorithm\nfci = Algorithm(\n    :FCI, ED(lattice, hilbert, (t₁, t₂, λ₂, t₃, V₁, V₂), quantumnumber, boundary);\n    parameters=parameters, map=map\n)\n\n# define the boundary angles and number of energy levels to be computed\nnθ, nev = 5, 3\nθs = range(0, 2, nθ)\ndata = zeros(nθ, nev)\n\n# compute the energy levels with different twist boundary angles\nfor (i, θ) in enumerate(θs)\n    update!(fci; θ₁=θ)\n    data[i, :] = eigen(fci; nev=nev).values\nend\n\n# plot the result\nplot(θs, data; legend=false)","category":"page"},{"location":"examples/HeisenbergModel/","page":"Antiferromagnetic Heisenberg Model on square lattice","title":"Antiferromagnetic Heisenberg Model on square lattice","text":"CurrentModule = ExactDiagonalization","category":"page"},{"location":"examples/HeisenbergModel/#Antiferromagnetic-Heisenberg-Model-on-square-lattice","page":"Antiferromagnetic Heisenberg Model on square lattice","title":"Antiferromagnetic Heisenberg Model on square lattice","text":"","category":"section"},{"location":"examples/HeisenbergModel/#Ground-state-energy","page":"Antiferromagnetic Heisenberg Model on square lattice","title":"Ground state energy","text":"","category":"section"},{"location":"examples/HeisenbergModel/","page":"Antiferromagnetic Heisenberg Model on square lattice","title":"Antiferromagnetic Heisenberg Model on square lattice","text":"The following codes could compute the ground state energy of the antiferromagnetic Heisenberg model on square lattice.","category":"page"},{"location":"examples/HeisenbergModel/","page":"Antiferromagnetic Heisenberg Model on square lattice","title":"Antiferromagnetic Heisenberg Model on square lattice","text":"using QuantumLattices\nusing ExactDiagonalization\nusing LinearAlgebra: eigen\n\n# define the unitcell of the square lattice\nunitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])\n\n# define a finite 4×4 cluster of the square lattice with open boundary condition\nlattice = Lattice(unitcell, (4, 4))\n\n# define the Hilbert space (spin-1/2)\nhilbert = Hilbert(Spin{1//2}(), length(lattice))\n\n# define the quantum number of the sub-Hilbert space in which the computation to be carried out\n# for the ground state, Sz=0\nquantumnumber = 𝕊ᶻ(0)\n\n# define the antiferromagnetic Heisenberg term on the nearest neighbor\nJ = Heisenberg(:J, 1.0, 1)\n\n# define the exact diagonalization algorithm for the antiferromagnetic Heisenberg model\ned = ED(lattice, hilbert, J, quantumnumber)\n\n# find the ground state and its energy\neigensystem = eigen(ed; nev=1)\n\n# Ground state energy should be -9.189207065192935\nprint(eigensystem.values)","category":"page"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"CurrentModule = ExactDiagonalization","category":"page"},{"location":"examples/Introduction/#examples","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Here are some examples to illustrate how this package could be used.","category":"page"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Pages = [\n        \"HubbardModel.md\",\n        \"HeisenbergModel.md\",\n        \"FractionalChernInsulatorOfHardCoreBosons.md\",\n        ]\nDepth = 2","category":"page"},{"location":"#ExactDiagonalization","page":"Home","title":"ExactDiagonalization","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: CI) (Image: codecov) (Image: ) (Image: ) (Image: 996.icu) (Image: LICENSE) (Image: LICENSE) (Image: Code Style: Blue) (Image: ColPrac: Contributor's Guide on Collaborative Practices for Community Packages)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Julia package for the exact diagonalization method in condensed matter physics.","category":"page"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Based on the symbolic operator representation of a quantum lattice system in condensed matter physics that is generated by the package QuantumLattices, exact diagonalization method is implemented for fermionic, hard-core-bosonic and spin systems.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In Julia v1.8+, please type ] in the REPL to use the package mode, then type this command:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add ExactDiagonalization","category":"page"},{"location":"#Getting-Started","page":"Home","title":"Getting Started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Examples of exact diagonalization method for quantum lattice system","category":"page"},{"location":"#Note","page":"Home","title":"Note","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Due to the fast development of this package, releases with different minor version numbers are not guaranteed to be compatible with previous ones before the release of v1.0.0. Comments are welcomed in the issues.","category":"page"},{"location":"#Contact","page":"Home","title":"Contact","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"waltergu1989@gmail.com","category":"page"},{"location":"#Python-counterpart","page":"Home","title":"Python counterpart","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"HamiltonianPy: in fact, the authors of this Julia package worked on the python package at first and only turned to Julia later.","category":"page"},{"location":"manul/","page":"Manual","title":"Manual","text":"CurrentModule = ExactDiagonalization","category":"page"},{"location":"manul/#Core-of-ExactDiagonalization","page":"Manual","title":"Core of ExactDiagonalization","text":"","category":"section"},{"location":"manul/","page":"Manual","title":"Manual","text":"Modules = [ExactDiagonalization]","category":"page"},{"location":"manul/#ExactDiagonalization.edtimer","page":"Manual","title":"ExactDiagonalization.edtimer","text":"const edtimer = TimerOutput()\n\nDefault shared timer for all exact diagonalization methods.\n\n\n\n\n\n","category":"constant"},{"location":"manul/#ExactDiagonalization.AbelianBases","page":"Manual","title":"ExactDiagonalization.AbelianBases","text":"AbelianBases{A<:Abelian, N} <: Sector\n\nA set of Abelian bases, that is, a set of bases composed from the product of local Abelian Graded spaces.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.AbelianBases-Union{Tuple{AbstractVector{Int64}}, Tuple{N}, Tuple{AbstractVector{Int64}, NTuple{N, AbstractVector{Int64}}}} where N","page":"Manual","title":"ExactDiagonalization.AbelianBases","text":"AbelianBases(locals::AbstractVector{Int}, partition::NTuple{N, AbstractVector{Int}}=partition(length(locals))) where N\n\nConstruct a set of spin bases that subjects to no quantum number conservation.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.AbelianBases-Union{Tuple{A}, Tuple{N}, Tuple{Array{QuantumLattices.QuantumNumbers.AbelianGradedSpace{A}, 1}, A}, Tuple{Array{QuantumLattices.QuantumNumbers.AbelianGradedSpace{A}, 1}, A, NTuple{N, AbstractVector{Int64}}}} where {N, A<:QuantumLattices.QuantumNumbers.AbelianQuantumNumber}","page":"Manual","title":"ExactDiagonalization.AbelianBases","text":"AbelianBases(locals::Vector{Graded{A}}, quantumnumber::A, partition::NTuple{N, AbstractVector{Int}}=partition(length(locals))) where {N, A<:Abelian}\n\nConstruct a set of spin bases that preserves a certain symmetry specified by the corresponding quantum number.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.BinaryBases","page":"Manual","title":"ExactDiagonalization.BinaryBases","text":"BinaryBases{A<:Abelian, B<:BinaryBasis, T<:AbstractVector{B}} <: Sector\n\nA set of binary bases.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.BinaryBases-Tuple{Any, Any, ℕ ⊠ 𝕊ᶻ}","page":"Manual","title":"ExactDiagonalization.BinaryBases","text":"BinaryBases(spindws, spinups, spinfulparticle::Abelian[ℕ ⊠ 𝕊ᶻ])\nBinaryBases(spindws, spinups, spinfulparticle::Abelian[𝕊ᶻ ⊠ ℕ])\n\nConstruct a set of binary bases that preserves both the particle number and the spin z-component conservation.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.BinaryBases-Tuple{Any, Any, 𝕊ᶻ}","page":"Manual","title":"ExactDiagonalization.BinaryBases","text":"BinaryBases(spindws, spinups, sz::𝕊ᶻ)\n\nConstruct a set of binary bases that preserves the spin z-component but not the particle number conservation.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.BinaryBases-Tuple{Integer, ℕ}","page":"Manual","title":"ExactDiagonalization.BinaryBases","text":"BinaryBases(states, particle::ℕ)\nBinaryBases(nstate::Integer, particle::ℕ)\n\nConstruct a set of binary bases that preserves the particle number conservation.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.BinaryBases-Tuple{Integer}","page":"Manual","title":"ExactDiagonalization.BinaryBases","text":"BinaryBases(states)\nBinaryBases(nstate::Integer)\n\nConstruct a set of binary bases that subjects to no quantum number conservation.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.BinaryBasis","page":"Manual","title":"ExactDiagonalization.BinaryBasis","text":"BinaryBasis{I<:Unsigned}\n\nBinary basis represented by an unsigned integer.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.BinaryBasis-Tuple{Any}","page":"Manual","title":"ExactDiagonalization.BinaryBasis","text":"BinaryBasis(states; filter=index->true)\nBinaryBasis{I}(states; filter=index->true) where {I<:Unsigned}\n\nConstruct a binary basis with the given occupied orbitals.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.BinaryBasisRange","page":"Manual","title":"ExactDiagonalization.BinaryBasisRange","text":"BinaryBasisRange{I<:Unsigned} <: VectorSpace{BinaryBasis{I}}\n\nA continuous range of binary basis from 0 to 2^n-1.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.ED","page":"Manual","title":"ExactDiagonalization.ED","text":"ED(\n    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, targetspace::TargetSpace=TargetSpace(hilbert), boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);\n    neighbors::Union{Int, Neighbors}=nneighbor(terms), timer::TimerOutput=edtimer, delay::Bool=false\n)\n\nConstruct the exact diagonalization method for a quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.ED-2","page":"Manual","title":"ExactDiagonalization.ED","text":"ED(system::Generator{<:Operators}, targetspace::TargetSpace, dtype::Type{<:Number}=scalartype(system); timer::TimerOutput=edtimer, delay::Bool=false)\nED(lattice::Union{AbstractLattice, Nothing}, system::Generator{<:Operators}, targetspace::TargetSpace, dtype::Type{<:Number}=scalartype(system); timer::TimerOutput=edtimer, delay::Bool=false)\n\nConstruct the exact diagonalization method for a quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.ED-3","page":"Manual","title":"ExactDiagonalization.ED","text":"ED(\n    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);\n    neighbors::Union{Int, Neighbors}=nneighbor(terms), timer::TimerOutput=edtimer, delay::Bool=false\n)\n\nConstruct the exact diagonalization method for a quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.ED-4","page":"Manual","title":"ExactDiagonalization.ED","text":"ED{K<:EDKind, L<:Union{AbstractLattice, Nothing}, S<:Generator{<:Operators}, M<:EDMatrixization, H<:CategorizedGenerator{<:OperatorSum{<:EDMatrix}}} <: Frontend\n\nExact diagonalization method of a quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.EDEigen","page":"Manual","title":"ExactDiagonalization.EDEigen","text":"EDEigen{V<:Number, T<:Number, S<:Sector} <: Factorization{T}\n\nEigen decomposition in exact diagonalization method.\n\nCompared to the usual eigen decomposition Eigen, EDEigen contains a :sectors attribute to store the sectors of Hilbert space in which the eigen values and eigen vectors are computed. Furthermore, given that in different sectors the dimensions of the sub-Hilbert spaces can also be different, the :vectors attribute of EDEigen is a vector of vector instead of a matrix.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.EDKind","page":"Manual","title":"ExactDiagonalization.EDKind","text":"EDKind{K}\n\nKind of the exact diagonalization method applied to a quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.EDKind-Tuple{Type{<:QuantumLattices.QuantumSystems.FockIndex}}","page":"Manual","title":"ExactDiagonalization.EDKind","text":"EDKind(::Type{<:FockIndex})\n\nKind of the exact diagonalization method applied to a canonical quantum Fock lattice system.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.EDKind-Tuple{Type{<:QuantumLattices.QuantumSystems.SpinIndex}}","page":"Manual","title":"ExactDiagonalization.EDKind","text":"EDKind(::Type{<:SpinIndex})\n\nKind of the exact diagonalization method applied to a canonical quantum spin lattice system.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.EDMatrix","page":"Manual","title":"ExactDiagonalization.EDMatrix","text":"EDMatrix{M<:SparseMatrixCSC, S<:Sector} <: OperatorPack{M, Tuple{S, S}}\n\nMatrix representation of quantum operators between a ket Hilbert space and a bra Hilbert space.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.EDMatrix-Tuple{SparseArrays.SparseMatrixCSC, Sector}","page":"Manual","title":"ExactDiagonalization.EDMatrix","text":"EDMatrix(m::SparseMatrixCSC, sector::Sector)\nEDMatrix(m::SparseMatrixCSC, braket::NTuple{2, Sector})\n\nConstruct a matrix representation when\n\nthe ket and bra Hilbert spaces share the same bases;\nthe ket and bra Hilbert spaces may be different.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.EDMatrixization","page":"Manual","title":"ExactDiagonalization.EDMatrixization","text":"EDMatrixization{D<:Number, S<:Sector, T<:AbstractDict} <: Matrixization\n\nMatrixization of a quantum lattice system on a target Hilbert space.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.EDMatrixization-Union{Tuple{TargetSpace}, Tuple{D}} where D<:Number","page":"Manual","title":"ExactDiagonalization.EDMatrixization","text":"EDMatrixization{D}(target::TargetSpace) where {D<:Number}\n\nConstruct a matrixization.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.Sector","page":"Manual","title":"ExactDiagonalization.Sector","text":"Sector(hilbert::Hilbert{<:Fock}, basistype::Type{<:Unsigned}=UInt) -> BinaryBases\nSector(hilbert::Hilbert{<:Fock}, quantumnumber::ℕ, basistype::Type{<:Unsigned}=UInt; table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))) -> BinaryBases\nSector(hilbert::Hilbert{<:Fock}, quantumnumber::Union{𝕊ᶻ, Abelian[ℕ ⊠ 𝕊ᶻ], Abelian[𝕊ᶻ ⊠ ℕ]}, basistype::Type{<:Unsigned}=UInt; table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))) -> BinaryBases\n\nConstruct the binary bases of a Hilbert space with the specified quantum number.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.Sector-2","page":"Manual","title":"ExactDiagonalization.Sector","text":"Sector\n\nA sector of the Hilbert space which forms the bases of an irreducible representation of the Hamiltonian of a quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.Sector-3","page":"Manual","title":"ExactDiagonalization.Sector","text":"Sector(hilbert::Hilbert{<:Spin}, partition::Tuple{Vararg{AbstractVector{Int}}}=partition(length(hilbert)); table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))) -> AbelianBases\nSector(hilbert::Hilbert{<:Spin}, quantumnumber::Abelian, partition::Tuple{Vararg{AbstractVector{Int}}}=partition(length(hilbert)); table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))) -> AbelianBases\n\nConstruct the Abelian bases of a spin Hilbert space with the specified quantum number.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.SectorFilter","page":"Manual","title":"ExactDiagonalization.SectorFilter","text":"SectorFilter{S} <: LinearTransformation\n\nFilter the target bra and ket Hilbert spaces.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.TargetSpace","page":"Manual","title":"ExactDiagonalization.TargetSpace","text":"TargetSpace{S<:Sector, T<:AbstractDict} <: VectorSpace{S}\n\nTarget Hilbert space in which the exact diagonalization method is performed, which could be the direct sum of several sectors.\n\n\n\n\n\n","category":"type"},{"location":"manul/#ExactDiagonalization.TargetSpace-Tuple{QuantumLattices.DegreesOfFreedom.Hilbert, Vararg{Any}}","page":"Manual","title":"ExactDiagonalization.TargetSpace","text":"TargetSpace(hilbert::Hilbert, args...)\nTargetSpace(hilbert::Hilbert, table::AbstractDict, args...)\nTargetSpace(hilbert::Hilbert, quantumnumbers::OneOrMore{Abelian}, args...)\n\nConstruct a target space from the total Hilbert space and the associated quantum numbers.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.TargetSpace-Union{Tuple{A}, Tuple{N}, Tuple{QuantumLattices.DegreesOfFreedom.Hilbert{<:QuantumLattices.QuantumSystems.Spin}, Union{Tuple{A, Vararg{A}}, A}, AbstractDict}, Tuple{QuantumLattices.DegreesOfFreedom.Hilbert{<:QuantumLattices.QuantumSystems.Spin}, Union{Tuple{A, Vararg{A}}, A}, AbstractDict, NTuple{N, AbstractVector{Int64}}}} where {N, A<:QuantumLattices.QuantumNumbers.AbelianQuantumNumber}","page":"Manual","title":"ExactDiagonalization.TargetSpace","text":"TargetSpace(hilbert::Hilbert{<:Spin}, quantumnumbers::OneOrMore{Abelian}, table::AbstractDict, partition::NTuple{N, AbstractVector{Int}}=partition(length(hilbert))) where N\n\nConstruct a target space from the total Hilbert space and the associated quantum numbers.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.TargetSpace-Union{Tuple{A}, Tuple{QuantumLattices.DegreesOfFreedom.Hilbert{<:QuantumLattices.QuantumSystems.Fock}, Union{Tuple{A, Vararg{A}}, A}, AbstractDict}, Tuple{QuantumLattices.DegreesOfFreedom.Hilbert{<:QuantumLattices.QuantumSystems.Fock}, Union{Tuple{A, Vararg{A}}, A}, AbstractDict, Type{<:Unsigned}}} where A<:QuantumLattices.QuantumNumbers.AbelianQuantumNumber","page":"Manual","title":"ExactDiagonalization.TargetSpace","text":"TargetSpace(hilbert::Hilbert{<:Fock}, quantumnumbers::OneOrMore{Abelian}, table::AbstractDict, basistype::Type{<:Unsigned}=UInt)\n\nConstruct a target space from the total Hilbert space and the associated quantum numbers.\n\n\n\n\n\n","category":"method"},{"location":"manul/#QuantumLattices.DegreesOfFreedom.Metric-Tuple{EDKind{:Abelian}, QuantumLattices.DegreesOfFreedom.Hilbert{<:QuantumLattices.QuantumSystems.Spin}}","page":"Manual","title":"QuantumLattices.DegreesOfFreedom.Metric","text":"Metric(::EDKind{:Abelian}, ::Hilbert{<:Spin}) -> OperatorIndexToTuple\n\nGet the index-to-tuple metric for a canonical quantum spin lattice system.\n\n\n\n\n\n","category":"method"},{"location":"manul/#QuantumLattices.DegreesOfFreedom.Metric-Tuple{EDKind{:Binary}, QuantumLattices.DegreesOfFreedom.Hilbert{<:QuantumLattices.QuantumSystems.Fock}}","page":"Manual","title":"QuantumLattices.DegreesOfFreedom.Metric","text":"Metric(::EDKind{:Binary}, ::Hilbert{<:Fock}) -> OperatorIndexToTuple\n\nGet the index-to-tuple metric for a canonical quantum Fock lattice system.\n\n\n\n\n\n","category":"method"},{"location":"manul/#QuantumLattices.QuantumNumbers.Abelian-Tuple{AbelianBases}","page":"Manual","title":"QuantumLattices.QuantumNumbers.Abelian","text":"Abelian(bs::AbelianBases)\n\nGet the quantum number of a set of spin bases.\n\n\n\n\n\n","category":"method"},{"location":"manul/#QuantumLattices.QuantumNumbers.Abelian-Tuple{BinaryBases}","page":"Manual","title":"QuantumLattices.QuantumNumbers.Abelian","text":"Abelian(bs::BinaryBases)\n\nGet the Abelian quantum number of a set of binary bases.\n\n\n\n\n\n","category":"method"},{"location":"manul/#QuantumLattices.QuantumNumbers.Graded-Union{Tuple{QuantumLattices.QuantumSystems.Spin}, Tuple{ℤ₁}} where ℤ₁","page":"Manual","title":"QuantumLattices.QuantumNumbers.Graded","text":"Graded{ℤ₁}(spin::Spin)\nGraded{𝕊ᶻ}(spin::Spin)\n\nDecompose a local spin space into an Abelian graded space that preserves 1) no symmetry, and 2) spin-z component symmetry.\n\n\n\n\n\n","category":"method"},{"location":"manul/#Base.count-Tuple{BinaryBasis}","page":"Manual","title":"Base.count","text":"count(basis::BinaryBasis) -> Int\ncount(basis::BinaryBasis, start::Integer, stop::Integer) -> Int\n\nCount the number of occupied single-particle states for a binary basis.\n\n\n\n\n\n","category":"method"},{"location":"manul/#Base.isone-Tuple{BinaryBasis, Integer}","page":"Manual","title":"Base.isone","text":"isone(basis::BinaryBasis, state::Integer) -> Bool\n\nJudge whether the specified single-particle state is occupied for a binary basis.\n\n\n\n\n\n","category":"method"},{"location":"manul/#Base.iszero-Tuple{BinaryBasis, Integer}","page":"Manual","title":"Base.iszero","text":"iszero(basis::BinaryBasis, state::Integer) -> Bool\n\nJudge whether the specified single-particle state is unoccupied for a binary basis.\n\n\n\n\n\n","category":"method"},{"location":"manul/#Base.iterate","page":"Manual","title":"Base.iterate","text":"iterate(basis::BinaryBasis)\niterate(basis::BinaryBasis, state)\n\nIterate over the numbers of the occupied single-particle orbitals.\n\n\n\n\n\n","category":"function"},{"location":"manul/#Base.match-Tuple{Sector, Sector}","page":"Manual","title":"Base.match","text":"match(sector₁::Sector, sector₂::Sector) -> Bool\n\nJudge whether two sectors match each other, that is, whether they can be used together as the bra and ket spaces.\n\n\n\n\n\n","category":"method"},{"location":"manul/#Base.one-Tuple{BinaryBasis, Integer}","page":"Manual","title":"Base.one","text":"one(basis::BinaryBasis, state::Integer) -> BinaryBasis\n\nGet a new binary basis with the specified single-particle state occupied.\n\n\n\n\n\n","category":"method"},{"location":"manul/#Base.range-Tuple{AbelianBases{ℤ₁}}","page":"Manual","title":"Base.range","text":"range(bs::AbelianBases) -> AbstractVector{Int}\n\nGet the range of the target sector of an AbelianBases in the direct product base.\n\n\n\n\n\n","category":"method"},{"location":"manul/#Base.zero-Tuple{BinaryBasis, Integer}","page":"Manual","title":"Base.zero","text":"zero(basis::BinaryBasis, state::Integer) -> BinaryBasis\n\nGet a new binary basis with the specified single-particle state unoccupied.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.basistype-Tuple{Integer}","page":"Manual","title":"ExactDiagonalization.basistype","text":"basistype(i::Integer)\nbasistype(::Type{I}) where {I<:Integer}\n\nGet the binary basis type corresponding to an integer or a type of an integer.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.productable-Tuple{Sector, Sector}","page":"Manual","title":"ExactDiagonalization.productable","text":"productable(sector₁::Sector, sector₂::Sector) -> Bool\n\nJudge whether two sectors could be direct producted.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.release!-Tuple{ED}","page":"Manual","title":"ExactDiagonalization.release!","text":"release!(ed::ED; gc::Bool=true) -> ED\nrelease!(ed::Algorithm{<:ED}; gc::Bool=true) -> Algorithm{<:ED}\n\nRelease the memory source used in preparing the matrix representation. If gc is true, call the garbage collection immediately.\n\n\n\n\n\n","category":"method"},{"location":"manul/#ExactDiagonalization.sumable-Tuple{Sector, Sector}","page":"Manual","title":"ExactDiagonalization.sumable","text":"sumable(sector₁::Sector, sector₂::Sector) -> Bool\n\nJudge whether two sectors could be direct summed.\n\n\n\n\n\n","category":"method"},{"location":"manul/#LinearAlgebra.eigen-Tuple{ED, Vararg{Union{Sector, QuantumLattices.QuantumNumbers.AbelianQuantumNumber}}}","page":"Manual","title":"LinearAlgebra.eigen","text":"eigen(ed::ED, sectors::Union{Abelian, Sector}...; timer::TimerOutput=edtimer, kwargs...) -> EDEigen\neigen(ed::Algorithm{<:ED}, sectors::Union{Abelian, Sector}...; kwargs...) -> EDEigen\n\nSolve the eigen problem by the restarted Lanczos method provided by the Arpack package.\n\n\n\n\n\n","category":"method"},{"location":"manul/#LinearAlgebra.eigen-Tuple{EDMatrix}","page":"Manual","title":"LinearAlgebra.eigen","text":"eigen(m::EDMatrix; nev::Int=1, which::Symbol=:SR, tol::Real=1e-12, maxiter::Int=300, v₀::Union{AbstractVector{<:Number}, Int}=dimension(m.bra), krylovdim::Int=max(20, 2*nev+1), verbosity::Int=0) -> EDEigen\n\nSolve the eigen problem by the restarted Lanczos method provided by the KrylovKit package.\n\n\n\n\n\n","category":"method"},{"location":"manul/#LinearAlgebra.eigen-Tuple{QuantumLattices.QuantumOperators.OperatorSum{<:EDMatrix}}","page":"Manual","title":"LinearAlgebra.eigen","text":"eigen(\n    ms::OperatorSum{<:EDMatrix};\n    nev::Int=1,\n    which::Symbol=:SR,\n    tol::Real=1e-12,\n    maxiter::Int=300,\n    v₀::Union{Dict{<:Abelian, <:Union{AbstractVector{<:Number}, Int}}, Dict{<:Sector, <:Union{AbstractVector{<:Number}, Int}}}=Dict(Abelian(m.ket)=>dimension(m.ket) for m in ms),\n    krylovdim::Int=max(20, 2*nev+1),\n    verbosity::Int=0,\n    timer::TimerOutput=edtimer\n) -> EDEigen\n\nSolve the eigen problem by the restarted Lanczos method provided by the Arpack package.\n\n\n\n\n\n","category":"method"},{"location":"manul/#QuantumLattices.:⊕-Tuple{TargetSpace, Vararg{Sector}}","page":"Manual","title":"QuantumLattices.:⊕","text":"⊕(target::TargetSpace, sectors::Sector...) -> TargetSpace\n\nGet the direct sum of a target space with several sectors.\n\n\n\n\n\n","category":"method"},{"location":"manul/#QuantumLattices.:⊗-Tuple{BinaryBases, BinaryBases}","page":"Manual","title":"QuantumLattices.:⊗","text":"⊗(bs₁::BinaryBases, bs₂::BinaryBases) -> BinaryBases\n\nGet the direct product of two sets of binary bases.\n\n\n\n\n\n","category":"method"},{"location":"manul/#QuantumLattices.:⊗-Tuple{BinaryBasis, BinaryBasis}","page":"Manual","title":"QuantumLattices.:⊗","text":"⊗(basis₁::BinaryBasis, basis₂::BinaryBasis) -> BinaryBasis\n\nGet the direct product of two binary bases.\n\n\n\n\n\n","category":"method"},{"location":"manul/#QuantumLattices.:⊠-Tuple{BinaryBases, QuantumLattices.QuantumNumbers.AbelianQuantumNumber}","page":"Manual","title":"QuantumLattices.:⊠","text":"⊠(bs::BinaryBases, another::Abelian) -> BinaryBases\n⊠(another::Abelian, bs::BinaryBases) -> BinaryBases\n\nDeligne tensor product the quantum number of a set of binary bases with another quantum number.\n\n\n\n\n\n","category":"method"},{"location":"manul/#QuantumLattices.DegreesOfFreedom.partition-Tuple{Int64}","page":"Manual","title":"QuantumLattices.DegreesOfFreedom.partition","text":"partition(n::Int) -> NTuple{2, Vector{Int}}\n\nGet the default partition of n local Hilbert spaces.\n\n\n\n\n\n","category":"method"},{"location":"manul/#QuantumLattices.Frameworks.prepare!-Tuple{ED}","page":"Manual","title":"QuantumLattices.Frameworks.prepare!","text":"prepare!(ed::ED; timer::TimerOutput=edtimer) -> ED\nprepare!(ed::Algorithm{<:ED}) -> Algorithm{<:ED}\n\nPrepare the matrix representation.\n\n\n\n\n\n","category":"method"},{"location":"manul/#QuantumLattices.QuantumOperators.matrix","page":"Manual","title":"QuantumLattices.QuantumOperators.matrix","text":"matrix(index::OperatorIndex, graded::Graded, dtype::Type{<:Number}=ComplexF64) -> Matrix{dtype}\n\nGet the matrix representation of an OperatorIndex on an Abelian graded space.\n\n\n\n\n\n","category":"function"},{"location":"manul/#QuantumLattices.QuantumOperators.matrix-2","page":"Manual","title":"QuantumLattices.QuantumOperators.matrix","text":"matrix(op::Operator, braket::NTuple{2, AbelianBases}, table::AbstractDict, dtype=valtype(op); kwargs...) -> SparseMatrixCSC{dtype, Int}\n\nGet the CSC-formed sparse matrix representation of an operator.\n\nHere, table specifies the order of the operator indexes.\n\n\n\n\n\n","category":"function"},{"location":"manul/#QuantumLattices.QuantumOperators.matrix-3","page":"Manual","title":"QuantumLattices.QuantumOperators.matrix","text":"matrix(op::Operator, braket::NTuple{2, BinaryBases}, table::AbstractDict, dtype=valtype(op); kwargs...) -> SparseMatrixCSC{dtype, Int}\n\nGet the CSC-formed sparse matrix representation of an operator.\n\nHere, table specifies the order of the operator indexes.\n\n\n\n\n\n","category":"function"},{"location":"manul/#QuantumLattices.QuantumOperators.matrix-4","page":"Manual","title":"QuantumLattices.QuantumOperators.matrix","text":"matrix(index::SpinIndex, graded::Graded, dtype::Type{<:Number}=ComplexF64) -> Matrix{dtype}\n\nGet the matrix representation of a SpinIndex on an Abelian graded space.\n\n\n\n\n\n","category":"function"},{"location":"manul/#QuantumLattices.QuantumOperators.matrix-5","page":"Manual","title":"QuantumLattices.QuantumOperators.matrix","text":"matrix(ops::Operators, braket::NTuple{2, Sector}, table::AbstractDict, dtype=scalartype(ops)) -> SparseMatrixCSC{dtype, Int}\n\nGet the CSC-formed sparse matrix representation of a set of operators.\n\nHere, table specifies the order of the operator indexes.\n\n\n\n\n\n","category":"function"},{"location":"manul/#QuantumLattices.QuantumOperators.matrix-Tuple{ED}","page":"Manual","title":"QuantumLattices.QuantumOperators.matrix","text":"matrix(ed::ED, sectors::Union{Abelian, Sector}...; timer::TimerOutput=edtimer, kwargs...) -> OperatorSum{<:EDMatrix}\nmatrix(ed::Algorithm{<:ED}, sectors::Union{Abelian, Sector}...; kwargs...) -> OperatorSum{<:EDMatrix}\n\nGet the sparse matrix representation of a quantum lattice system in the target space.\n\n\n\n\n\n","category":"method"},{"location":"examples/HubbardModel/","page":"Fermi Hubbard Model on square lattice","title":"Fermi Hubbard Model on square lattice","text":"CurrentModule = ExactDiagonalization","category":"page"},{"location":"examples/HubbardModel/#Fermi-Hubbard-Model-on-square-lattice","page":"Fermi Hubbard Model on square lattice","title":"Fermi Hubbard Model on square lattice","text":"","category":"section"},{"location":"examples/HubbardModel/#Ground-state-energy","page":"Fermi Hubbard Model on square lattice","title":"Ground state energy","text":"","category":"section"},{"location":"examples/HubbardModel/","page":"Fermi Hubbard Model on square lattice","title":"Fermi Hubbard Model on square lattice","text":"The following codes could compute the ground state energy of the Fermi Hubbard model on square lattice.","category":"page"},{"location":"examples/HubbardModel/","page":"Fermi Hubbard Model on square lattice","title":"Fermi Hubbard Model on square lattice","text":"using QuantumLattices\nusing ExactDiagonalization\nusing LinearAlgebra: eigen\n\n# define the unitcell of the square lattice\nunitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])\n\n# define a finite 3×4 cluster of the square lattice with open boundary condition\nlattice = Lattice(unitcell, (3, 4))\n\n# define the Hilbert space (single-orbital spin-1/2 complex fermion)\nhilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(lattice))\n\n# define the quantum number of the sub-Hilbert space in which the computation to be carried out\n# here the particle number is set to be `length(lattice)` and Sz is set to be 0\nquantumnumber = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)\n\n# define the terms, i.e. the nearest-neighbor hopping and the Hubbard interaction\nt = Hopping(:t, -1.0, 1)\nU = Hubbard(:U, 8.0)\n\n# define the exact diagonalization algorithm for the Fermi Hubbard model\ned = ED(lattice, hilbert, (t, U), quantumnumber)\n\n# find the ground state and its energy\neigensystem = eigen(ed; nev=1)\n\n# Ground state energy should be -4.913259209075605\nprint(eigensystem.values)","category":"page"}]
}