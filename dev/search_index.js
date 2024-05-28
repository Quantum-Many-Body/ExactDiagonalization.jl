var documenterSearchIndex = {"docs":
[{"location":"examples/FractionalChernInsulatorOfHardCoreBosons/","page":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","title":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","text":"CurrentModule = ExactDiagonalization","category":"page"},{"location":"examples/FractionalChernInsulatorOfHardCoreBosons/#Fractional-Chern-Insulator-of-Hard-core-Bosons-on-honeycomb-lattice","page":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","title":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","text":"","category":"section"},{"location":"examples/FractionalChernInsulatorOfHardCoreBosons/#Many-body-energy-levels-as-a-function-of-twisted-boundary-conditions","page":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","title":"Many-body energy levels as a function of twisted boundary conditions","text":"","category":"section"},{"location":"examples/FractionalChernInsulatorOfHardCoreBosons/","page":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","title":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","text":"The following codes could compute the many-body energy levels as a function of twisted boundary conditions of hard-core bosons on honeycomb lattice with a nearly-flat Chern band.","category":"page"},{"location":"examples/FractionalChernInsulatorOfHardCoreBosons/","page":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","title":"Fractional Chern Insulator of Hard-core Bosons on honeycomb lattice","text":"using QuantumLattices\nusing ExactDiagonalization\nusing LinearAlgebra: eigen\nusing Plots: plot\n\n# unit cell of the honeycomb lattice\nunitcell = Lattice([0.0, 0.0], [0.0, √3/3]; vectors=[[1.0, 0.0], [0.5, √3/2]])\n\n# 4×3 cluster\nlattice = Lattice(unitcell, (4, 3), ('P', 'P'))\n\n# twisted boundary condition\nboundary = Boundary{(:θ₁, :θ₂)}([0.0, 0.0], lattice.vectors)\n\n# Hilbert space of hard-core bosons\nhilbert = Hilbert(Fock{:b}(1, 1), length(lattice))\n\n# Parameters of the model, with hoppings up to 3rd nearest neighbor and interactions up to 2nd\n# the 2nd nearest neighbor hopping is complex, similar to that of the Haldane model\n# with such parameters, the model host a nearly flat band with ±1 Chern number\nparameters = (\n    t=Complex(-1.0), t′=Complex(-0.6), t′′=Complex(0.58), φ=0.4, V₁=1.0, V₂=0.4, θ₁=0.0, θ₂=0.0\n)\nmap(parameters::NamedTuple) = (\n    t₁=parameters.t,\n    t₂=parameters.t′*cos(parameters.φ*pi),\n    λ₂=parameters.t′*sin(parameters.φ*pi),\n    t₃=parameters.t′′,\n    V₁=parameters.V₁,\n    V₂=parameters.V₂,\n    θ₁=parameters.θ₁,\n    θ₂=parameters.θ₂\n)\n\n# terms\nt₁ = Hopping(:t₁, map(parameters).t₁, 1; modulate=false)\nt₂ = Hopping(:t₂, map(parameters).t₂, 2; modulate=false)\nλ₂ = Hopping(:λ₂, map(parameters).λ₂, 2;\n    amplitude=bond::Bond->1im*cos(3*azimuth(rcoordinate(bond)))*(-1)^(bond[1].site%2),\n    modulate=false\n)\nt₃ = Hopping(:t₃, map(parameters).t₃, 3; modulate=false)\nV₁ = Coulomb(:V₁, map(parameters).V₁, 1)\nV₂ = Coulomb(:V₂, map(parameters).V₂, 2)\n\n# 1/4 filling of the model, i.e., half-filling of the lower flat Chern band\n# In such case, the ground state of the model is a bosonic fractional Chern insulator\n# see Y.-F. Wang, et al. PRL 107, 146803 (2011)\nquantumnumber = ParticleNumber(length(lattice)/4)\n\n# construct the algorithm\nfci = Algorithm(\n    :FCI, ED(lattice, hilbert, (t₁, t₂, λ₂, t₃, V₁, V₂), quantumnumber, boundary);\n    parameters=parameters, map=map\n)\n\n# define the boundary angles and number of energy levels to be computed\nnθ, nev = 5, 3\nθs = range(0, 2, nθ)\ndata = zeros(nθ, nev)\n\n# compute the energy levels with different twist boundary angles\nfor (i, θ) in enumerate(θs)\n    update!(fci; θ₁=θ)\n    data[i, :] = eigen(fci; nev=nev).values\nend\n\n# plot the result\nplot(θs, data; legend=false)","category":"page"},{"location":"examples/HeisenbergModel/","page":"Antiferromagnetic Heisenberg Model on square lattice","title":"Antiferromagnetic Heisenberg Model on square lattice","text":"CurrentModule = ExactDiagonalization","category":"page"},{"location":"examples/HeisenbergModel/#Antiferromagnetic-Heisenberg-Model-on-square-lattice","page":"Antiferromagnetic Heisenberg Model on square lattice","title":"Antiferromagnetic Heisenberg Model on square lattice","text":"","category":"section"},{"location":"examples/HeisenbergModel/#Ground-state-energy","page":"Antiferromagnetic Heisenberg Model on square lattice","title":"Ground state energy","text":"","category":"section"},{"location":"examples/HeisenbergModel/","page":"Antiferromagnetic Heisenberg Model on square lattice","title":"Antiferromagnetic Heisenberg Model on square lattice","text":"The following codes could compute the ground state energy of the antiferromagnetic Heisenberg model on square lattice.","category":"page"},{"location":"examples/HeisenbergModel/","page":"Antiferromagnetic Heisenberg Model on square lattice","title":"Antiferromagnetic Heisenberg Model on square lattice","text":"using QuantumLattices\nusing ExactDiagonalization\nusing LinearAlgebra: eigen\n\n# define the unitcell of the square lattice\nunitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])\n\n# define a finite 4×4 cluster of the square lattice with open boundary condition\nlattice = Lattice(unitcell, (4, 4))\n\n# define the Hilbert space (spin-1/2)\nhilbert = Hilbert(Spin{1//2}(), length(lattice))\n\n# define the quantum number of the sub-Hilbert space in which the computation to be carried out\n# for the ground state, Sz=0\nquantumnumber = Sz(0)\n\n# define the antiferromagnetic Heisenberg term on the nearest neighbor\nJ = Heisenberg(:J, 1.0, 1)\n\n# define the exact diagonalization algorithm for the antiferromagnetic Heisenberg model\ned = ED(lattice, hilbert, J, quantumnumber)\n\n# find the ground state and its energy\neigensystem = eigen(ed; nev=1)\n\n# Ground state energy should be -9.189207065192935\nprint(eigensystem.values)","category":"page"},{"location":"man/CanonicalSpinSystems/","page":"Canonical Spin Systems","title":"Canonical Spin Systems","text":"CurrentModule = ExactDiagonalization.CanonicalSpinSystems","category":"page"},{"location":"man/CanonicalSpinSystems/#Canonical-Spin-Systems","page":"Canonical Spin Systems","title":"Canonical Spin Systems","text":"","category":"section"},{"location":"man/CanonicalSpinSystems/","page":"Canonical Spin Systems","title":"Canonical Spin Systems","text":"Modules = [CanonicalSpinSystems]\nOrder = [:module, :constant, :type, :macro, :function]","category":"page"},{"location":"man/CanonicalSpinSystems/#ExactDiagonalization.CanonicalSpinSystems.SpinBases","page":"Canonical Spin Systems","title":"ExactDiagonalization.CanonicalSpinSystems.SpinBases","text":"SpinBases <: Sector\n\nA set of spin bases.\n\n\n\n\n\n","category":"type"},{"location":"man/CanonicalSpinSystems/#ExactDiagonalization.CanonicalSpinSystems.SpinBases-Tuple{Vector{<:Real}}","page":"Canonical Spin Systems","title":"ExactDiagonalization.CanonicalSpinSystems.SpinBases","text":"SpinBases(spins::Vector{<:Real})\nSpinBases(spins::Vector{<:Real}, partition::NTuple{N, AbstractVector{Int}}) where N\nSpinBases(spins::Vector{<:Real}, quantumnumber::Sz)\nSpinBases(spins::Vector{<:Real}, quantumnumber::Sz, partition::NTuple{N, AbstractVector{Int}}) where N\n\nConstruct a set of spin bases.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalSpinSystems/#ExactDiagonalization.EDCore.EDKind-Tuple{Type{<:QuantumLattices.DegreesOfFreedom.Hilbert{<:QuantumLattices.QuantumSystems.Spin}}}","page":"Canonical Spin Systems","title":"ExactDiagonalization.EDCore.EDKind","text":"EDKind(::Type{<:Hilbert{<:Spin}})\n\nThe kind of the exact diagonalization method applied to a canonical quantum spin lattice system.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalSpinSystems/#ExactDiagonalization.EDCore.Sector","page":"Canonical Spin Systems","title":"ExactDiagonalization.EDCore.Sector","text":"Sector(hilbert::Hilbert{<:Spin}, partition::Tuple{Vararg{AbstractVector{Int}}}=defaultpartition(length(hilbert)))\nSector(hilbert::Hilbert{<:Spin}, quantumnumber::Sz, partition::Tuple{Vararg{AbstractVector{Int}}}=defaultpartition(length(hilbert)))\n\nConstruct the spin bases of a Hilbert space with the specified quantum number.\n\n\n\n\n\n","category":"type"},{"location":"man/CanonicalSpinSystems/#ExactDiagonalization.EDCore.TargetSpace","page":"Canonical Spin Systems","title":"ExactDiagonalization.EDCore.TargetSpace","text":"TargetSpace(hilbert::Hilbert{<:Spin}, partition::Tuple{Vararg{AbstractVector{Int}}}=defaultpartition(length(hilbert)))\nTargetSpace(hilbert::Hilbert{<:Spin}, quantumnumbers::Union{Sz, Tuple{Sz, Vararg{Sz}}}, partition::Tuple{Vararg{AbstractVector{Int}}}=defaultpartition(length(hilbert)))\n\nConstruct a target space from the total Hilbert space and the associated quantum numbers.\n\n\n\n\n\n","category":"type"},{"location":"man/CanonicalSpinSystems/#QuantumLattices.DegreesOfFreedom.Metric-Tuple{EDKind{:SED}, QuantumLattices.DegreesOfFreedom.Hilbert{<:QuantumLattices.QuantumSystems.Spin}}","page":"Canonical Spin Systems","title":"QuantumLattices.DegreesOfFreedom.Metric","text":"Metric(::EDKind{:SED}, ::Hilbert{<:Spin}) -> OperatorUnitToTuple\n\nGet the index-to-tuple metric for a canonical quantum spin lattice system.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalSpinSystems/#QuantumLattices.QuantumNumbers.AbelianNumber-Tuple{SpinBases}","page":"Canonical Spin Systems","title":"QuantumLattices.QuantumNumbers.AbelianNumber","text":"AbelianNumber(bs::SpinBases)\n\nGet the quantum number of a set of spin bases.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalSpinSystems/#ExactDiagonalization.CanonicalSpinSystems.Pspincoherentstates-Union{Tuple{T}, Tuple{AbstractVector{T}, Dict{Vector{Int64}, Vector{Float64}}}} where T<:Number","page":"Canonical Spin Systems","title":"ExactDiagonalization.CanonicalSpinSystems.Pspincoherentstates","text":"Pspincoherentstates(scs::AbstractVector{T}, spins::Dict{Vector{Int}, Vector{Float64}}; N::Int=100) where {T<:Number}\n\nGet square of the Projectors of state \"scs\" onto spincoherentstates.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalSpinSystems/#ExactDiagonalization.CanonicalSpinSystems.spincoherentstates-Tuple{Matrix{Float64}}","page":"Canonical Spin Systems","title":"ExactDiagonalization.CanonicalSpinSystems.spincoherentstates","text":"spincoherentstates(structure::Matrix{Float64}) -> Matrix{Float64}\n\nGet the spin coherent states from the input spin structures specified by the polar and azimuth angles.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalSpinSystems/#ExactDiagonalization.CanonicalSpinSystems.structure_factor-Union{Tuple{T}, Tuple{QuantumLattices.Spatials.AbstractLattice, SpinBases, QuantumLattices.DegreesOfFreedom.Hilbert, AbstractVector{T}, Vector{Float64}}} where T<:Number","page":"Canonical Spin Systems","title":"ExactDiagonalization.CanonicalSpinSystems.structure_factor","text":"structure_factor(lattice::AbstractLattice, bs::SpinBases, hilbert::Hilbert, scs::AbstractVector{T}, k::Vector{Float64}) where {T<:Number} -> [SxSx(k), SySy(k), SzSz(k)]\nstructure_factor(lattice::AbstractLattice, bs::SpinBases, hilbert::Hilbert, scs::AbstractVector{T}; Nk::Int=60) where {T<:Number} -> Matrix(3, Nk, Nk)\n\nGet structure_factor of state \"scs\".\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalSpinSystems/#ExactDiagonalization.EDCore.sumable-Tuple{SpinBases, SpinBases}","page":"Canonical Spin Systems","title":"ExactDiagonalization.EDCore.sumable","text":"sumable(bs₁::SpinBases, bs₂::SpinBases) -> Bool\n\nJudge whether two sets of spin bases could be direct summed.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalSpinSystems/#QuantumLattices.QuantumOperators.matrix","page":"Canonical Spin Systems","title":"QuantumLattices.QuantumOperators.matrix","text":"matrix(op::Operator, braket::NTuple{2, SpinBases}, table, dtype=valtype(op)) -> SparseMatrixCSC{dtype, Int}\n\nGet the CSC-formed sparse matrix representation of an operator.\n\nHere, table specifies the order of the operator ids.\n\n\n\n\n\n","category":"function"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"CurrentModule = ExactDiagonalization","category":"page"},{"location":"examples/Introduction/#examples","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Here are some examples to illustrate how this package could be used.","category":"page"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Pages = [\n        \"HubbardModel.md\",\n        \"HeisenbergModel.md\",\n        \"FractionalChernInsulatorOfHardCoreBosons.md\",\n        ]\nDepth = 2","category":"page"},{"location":"man/CanonicalFockSystems/","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"Canonical Fermionic and Hard-core Bosonic Systems","text":"CurrentModule = ExactDiagonalization.CanonicalFockSystems","category":"page"},{"location":"man/CanonicalFockSystems/#Canonical-Fermionic-and-Hard-core-Bosonic-Systems","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"Canonical Fermionic and Hard-core Bosonic Systems","text":"","category":"section"},{"location":"man/CanonicalFockSystems/","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"Canonical Fermionic and Hard-core Bosonic Systems","text":"Modules = [CanonicalFockSystems]\nOrder = [:module, :constant, :type, :macro, :function]","category":"page"},{"location":"man/CanonicalFockSystems/#ExactDiagonalization.CanonicalFockSystems.BinaryBases","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"ExactDiagonalization.CanonicalFockSystems.BinaryBases","text":"BinaryBases{A<:AbelianNumber, B<:BinaryBasis, T<:AbstractVector{B}} <: Sector\n\nA set of binary bases.\n\n\n\n\n\n","category":"type"},{"location":"man/CanonicalFockSystems/#ExactDiagonalization.CanonicalFockSystems.BinaryBases-Tuple{Any, Integer}","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"ExactDiagonalization.CanonicalFockSystems.BinaryBases","text":"BinaryBases(states, nparticle::Integer)\nBinaryBases(nstate::Integer, nparticle::Integer)\nBinaryBases{A}(states, nparticle::Integer; kwargs...) where {A<:AbelianNumber}\nBinaryBases{A}(nstate::Integer, nparticle::Integer; kwargs...) where {A<:AbelianNumber}\n\nConstruct a set of binary bases that preserves the particle number conservation.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalFockSystems/#ExactDiagonalization.CanonicalFockSystems.BinaryBases-Tuple{Any}","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"ExactDiagonalization.CanonicalFockSystems.BinaryBases","text":"BinaryBases(states)\nBinaryBases(nstate::Integer)\nBinaryBases{A}(states) where {A<:AbelianNumber}\nBinaryBases{A}(nstate::Integer) where {A<:AbelianNumber}\n\nConstruct a set of binary bases that subject to no quantum number conservation.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalFockSystems/#ExactDiagonalization.CanonicalFockSystems.BinaryBasis","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"ExactDiagonalization.CanonicalFockSystems.BinaryBasis","text":"BinaryBasis{I<:Unsigned}\n\nBinary basis represented by an unsigned integer.\n\n\n\n\n\n","category":"type"},{"location":"man/CanonicalFockSystems/#ExactDiagonalization.CanonicalFockSystems.BinaryBasis-Tuple{Any}","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"ExactDiagonalization.CanonicalFockSystems.BinaryBasis","text":"BinaryBasis(states; filter=index->true)\nBinaryBasis{I}(states; filter=index->true) where {I<:Unsigned}\n\nConstruct a binary basis with the given occupied orbitals.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalFockSystems/#ExactDiagonalization.CanonicalFockSystems.BinaryBasisRange","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"ExactDiagonalization.CanonicalFockSystems.BinaryBasisRange","text":"BinaryBasisRange{I<:Unsigned} <: VectorSpace{BinaryBasis{I}}\n\nA continuous range of binary basis from 0 to 2^n-1.\n\n\n\n\n\n","category":"type"},{"location":"man/CanonicalFockSystems/#ExactDiagonalization.EDCore.EDKind-Tuple{Type{<:QuantumLattices.DegreesOfFreedom.Hilbert{<:QuantumLattices.QuantumSystems.Fock}}}","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"ExactDiagonalization.EDCore.EDKind","text":"EDKind(::Type{<:Hilbert{<:Fock}})\n\nThe kind of the exact diagonalization method applied to a canonical quantum Fock lattice system.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalFockSystems/#ExactDiagonalization.EDCore.Sector","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"ExactDiagonalization.EDCore.Sector","text":"Sector(hilbert::Hilbert{<:Fock}, basistype=UInt) -> BinaryBases\nSector(hilbert::Hilbert{<:Fock}, quantumnumber::ParticleNumber, basistype=UInt; table=Table(hilbert, Metric(EDKind(hilbert), hilbert))) -> BinaryBases\nSector(hilbert::Hilbert{<:Fock}, quantumnumber::SpinfulParticle, basistype=UInt; table=Table(hilbert, Metric(EDKind(hilbert), hilbert))) -> BinaryBases\n\nConstruct the binary bases of a Hilbert space with the specified quantum number.\n\n\n\n\n\n","category":"type"},{"location":"man/CanonicalFockSystems/#ExactDiagonalization.EDCore.TargetSpace","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"ExactDiagonalization.EDCore.TargetSpace","text":"TargetSpace(hilbert::Hilbert{<:Fock}, basistype=UInt)\nTargetSpace(hilbert::Hilbert{<:Fock}, quantumnumbers::Union{AbelianNumber, Tuple{AbelianNumber, Vararg{AbelianNumber}}}, basistype=UInt; table=Table(hilbert, Metric(EDKind(hilbert), hilbert)))\n\nConstruct a target space from the total Hilbert space and the associated quantum numbers.\n\n\n\n\n\n","category":"type"},{"location":"man/CanonicalFockSystems/#QuantumLattices.DegreesOfFreedom.Metric-Tuple{EDKind{:FED}, QuantumLattices.DegreesOfFreedom.Hilbert{<:QuantumLattices.QuantumSystems.Fock}}","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"QuantumLattices.DegreesOfFreedom.Metric","text":"Metric(::EDKind{:FED}, ::Hilbert{<:Fock}) -> OperatorUnitToTuple\n\nGet the index-to-tuple metric for a canonical quantum Fock lattice system.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalFockSystems/#QuantumLattices.QuantumNumbers.AbelianNumber-Tuple{BinaryBases}","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"QuantumLattices.QuantumNumbers.AbelianNumber","text":"AbelianNumber(bs::BinaryBases)\n\nGet the Abelian quantum number of a set of binary bases.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalFockSystems/#Base.count-Tuple{BinaryBasis}","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"Base.count","text":"count(basis::BinaryBasis) -> Int\ncount(basis::BinaryBasis, start::Integer, stop::Integer) -> Int\n\nCount the number of occupied single-particle states.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalFockSystems/#Base.isone-Tuple{BinaryBasis, Integer}","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"Base.isone","text":"isone(basis::BinaryBasis, state::Integer) -> Bool\n\nJudge whether the specified single-particle state is occupied for a basis.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalFockSystems/#Base.iszero-Tuple{BinaryBasis, Integer}","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"Base.iszero","text":"iszero(basis::BinaryBasis, state::Integer) -> Bool\n\nJudge whether the specified single-particle state is unoccupied for a basis.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalFockSystems/#Base.iterate","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"Base.iterate","text":"iterate(basis::BinaryBasis, state=nothing)\n\nIterate over the numbers of the occupied single-particle orbitals.\n\n\n\n\n\n","category":"function"},{"location":"man/CanonicalFockSystems/#Base.one-Tuple{BinaryBasis, Integer}","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"Base.one","text":"one(basis::BinaryBasis, state::Integer) -> BinaryBasis\n\nGet a new basis with the specified single-particle state occupied. \n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalFockSystems/#Base.zero-Tuple{BinaryBasis, Integer}","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"Base.zero","text":"zero(basis::BinaryBasis, state::Integer) -> BinaryBasis\n\nGet a new basis with the specified single-particle state unoccupied.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalFockSystems/#ExactDiagonalization.EDCore.productable-Union{Tuple{A₂}, Tuple{A₁}, Tuple{BinaryBases{A₁, B} where B<:BinaryBasis, BinaryBases{A₂, B} where B<:BinaryBasis}} where {A₁<:QuantumLattices.QuantumNumbers.AbelianNumber, A₂<:QuantumLattices.QuantumNumbers.AbelianNumber}","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"ExactDiagonalization.EDCore.productable","text":"productable(bs₁::BinaryBases, bs₂::BinaryBases) -> Bool\n\nJudge whether two sets of binary bases could be direct producted.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalFockSystems/#ExactDiagonalization.EDCore.sumable-Union{Tuple{A₂}, Tuple{A₁}, Tuple{BinaryBases{A₁, B} where B<:BinaryBasis, BinaryBases{A₂, B} where B<:BinaryBasis}} where {A₁<:QuantumLattices.QuantumNumbers.AbelianNumber, A₂<:QuantumLattices.QuantumNumbers.AbelianNumber}","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"ExactDiagonalization.EDCore.sumable","text":"sumable(bs₁::BinaryBases, bs₂::BinaryBases) -> Bool\n\nJudge whether two sets of binary bases could be direct summed.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalFockSystems/#QuantumLattices.:⊗-Tuple{BinaryBases, BinaryBases}","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"QuantumLattices.:⊗","text":"⊗(bs₁::BinaryBases, bs₂::BinaryBases) -> BinaryBases\n\nGet the direct product of two sets of binary bases.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalFockSystems/#QuantumLattices.:⊗-Tuple{BinaryBasis, BinaryBasis}","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"QuantumLattices.:⊗","text":"⊗(basis₁::BinaryBasis, basis₂::BinaryBasis) -> BinaryBasis\n\nGet the direct product of two binary bases.\n\n\n\n\n\n","category":"method"},{"location":"man/CanonicalFockSystems/#QuantumLattices.QuantumOperators.matrix","page":"Canonical Fermionic and Hard-core Bosonic Systems","title":"QuantumLattices.QuantumOperators.matrix","text":"matrix(op::Operator, braket::NTuple{2, BinaryBases}, table, dtype=valtype(op)) -> SparseMatrixCSC{dtype, Int}\n\nGet the CSC-formed sparse matrix representation of an operator.\n\nHere, table specifies the order of the operator ids.\n\n\n\n\n\n","category":"function"},{"location":"man/EDCore/","page":"Core of Exact Diagonalization","title":"Core of Exact Diagonalization","text":"CurrentModule = ExactDiagonalization.EDCore","category":"page"},{"location":"man/EDCore/#Core-of-Exact-Diagonalization","page":"Core of Exact Diagonalization","title":"Core of Exact Diagonalization","text":"","category":"section"},{"location":"man/EDCore/","page":"Core of Exact Diagonalization","title":"Core of Exact Diagonalization","text":"Modules = [EDCore]\nOrder = [:module, :constant, :type, :macro, :function]","category":"page"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.edtimer","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.edtimer","text":"const edtimer = TimerOutput()\n\nThe default shared timer for all exact diagonalization methods.\n\n\n\n\n\n","category":"constant"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.ED","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.ED","text":"ED(\n    lattice::AbstractLattice,\n    hilbert::Hilbert,\n    terms::Union{Term, Tuple{Term, Vararg{Term}}},\n    quantumnumbers::Union{AbelianNumber, Tuple{AbelianNumber, Vararg{AbelianNumber}}},\n    dtype::Type{<:Number}=commontype(terms),\n    boundary::Boundary=plain;\n    neighbors::Union{Nothing, Int, Neighbors}=nothing,\n    timer::TimerOutput=edtimer,\n)\n\nConstruct the exact diagonalization method for a quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.ED-2","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.ED","text":"ED(\n    lattice::AbstractLattice, hilbert::Hilbert, terms::Union{Term, Tuple{Term, Vararg{Term}}}, targetspace::TargetSpace=TargetSpace(hilbert), dtype::Type{<:Number}=commontype(terms), boundary::Boundary=plain;\n    neighbors::Union{Nothing, Int, Neighbors}=nothing, timer::TimerOutput=edtimer\n)\n\nConstruct the exact diagonalization method for a quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.ED-3","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.ED","text":"ED{K<:EDKind, L<:AbstractLattice, G<:OperatorGenerator, M<:Image} <: Frontend\n\nExact diagonalization method of a quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.EDEigen","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.EDEigen","text":"EDEigen{V<:Number, T<:Number, S<:Sector} <: Factorization{T}\n\nEigen decomposition in exact diagonalization method.\n\nCompared to the usual eigen decomposition Eigen, EDEigen contains a :sectors attribute to store the sectors of Hilbert space in which the eigen values and eigen vectors are computed. Furthermore, given that in different sectors the dimensions of the sub-Hilbert spaces can also be different, the :vectors attribute of EDEigen is a vector of vector instead of a matrix.\n\n\n\n\n\n","category":"type"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.EDKind","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.EDKind","text":"EDKind{K}\n\nThe kind of the exact diagonalization method applied to a quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.EDMatrix","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.EDMatrix","text":"EDMatrix{S<:Sector, M<:SparseMatrixCSC} <: OperatorPack{M, Tuple{S, S}}\n\nMatrix representation of quantum operators between a ket and a bra Hilbert space.\n\n\n\n\n\n","category":"type"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.EDMatrix-Tuple{Sector, SparseArrays.SparseMatrixCSC}","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.EDMatrix","text":"EDMatrix(sector::Sector, m::SparseMatrixCSC)\nEDMatrix(braket::NTuple{2, Sector}, m::SparseMatrixCSC)\n\nConstruct a matrix representation when\n\nthe ket and bra spaces share the same bases;\n\n2-3) the ket and bra spaces may be different.\n\n\n\n\n\n","category":"method"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.EDMatrixRepresentation","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.EDMatrixRepresentation","text":"EDMatrixRepresentation{D<:Number, S<:Sector, T} <: MatrixRepresentation\n\nExact matrix representation of a quantum lattice system on a target Hilbert space.\n\n\n\n\n\n","category":"type"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.EDMatrixRepresentation-Union{Tuple{D}, Tuple{TargetSpace, Any}} where D<:Number","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.EDMatrixRepresentation","text":"EDMatrixRepresentation{D}(target::TargetSpace, table) where {D<:Number}\n\nConstruct a exact matrix representation.\n\n\n\n\n\n","category":"method"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.Sector","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.Sector","text":"abstract type Sector <: OperatorUnit\n\nA sector of the Hilbert space which forms the bases of an irreducible representation of the Hamiltonian of a quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.SectorFilter","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.SectorFilter","text":"SectorFilter{S} <: LinearTransformation\n\nFilter the target bra and ket Hilbert spaces.\n\n\n\n\n\n","category":"type"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.TargetSpace","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.TargetSpace","text":"TargetSpace{S<:Sector} <: VectorSpace{S}\n\nThe target Hilbert space in which the exact diagonalization method is performed, which could be the direct sum of several sectors.\n\n\n\n\n\n","category":"type"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.TargetSpace-Tuple{QuantumLattices.DegreesOfFreedom.Hilbert}","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.TargetSpace","text":"TargetSpace(hilbert::Hilbert)\nTargetSpace(hilbert::Hilbert, quantumnumbers::Union{AbelianNumber, Tuple{AbelianNumber, Vararg{AbelianNumber}}})\n\nConstruct a target space from the total Hilbert space and the associated quantum numbers.\n\n\n\n\n\n","category":"method"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.TargetSpace-Tuple{Sector, Vararg{Sector}}","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.TargetSpace","text":"TargetSpace(sector::Sector, sectors::Sector...)\n\nConstruct a target space from sectors.\n\n\n\n\n\n","category":"method"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.productable","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.productable","text":"productable(sector₁::Sector, sector₂::Sector) -> Bool\n\nJudge whether two sectors could be direct producted.\n\n\n\n\n\n","category":"function"},{"location":"man/EDCore/#ExactDiagonalization.EDCore.sumable","page":"Core of Exact Diagonalization","title":"ExactDiagonalization.EDCore.sumable","text":"sumable(sector₁::Sector, sector₂::Sector) -> Bool\n\nJudge whether two sectors could be direct summed.\n\n\n\n\n\n","category":"function"},{"location":"man/EDCore/#LinearAlgebra.eigen-Tuple{ED, Vararg{Union{Sector, QuantumLattices.QuantumNumbers.AbelianNumber}}}","page":"Core of Exact Diagonalization","title":"LinearAlgebra.eigen","text":"eigen(ed::ED, sectors::Union{AbelianNumber, Sector}...; timer::TimerOutput=edtimer, kwargs...) -> EDEigen\neigen(ed::Algorithm{<:ED}, sectors::Union{AbelianNumber, Sector}...; kwargs...) -> EDEigen\n\nSolve the eigen problem by the restarted Lanczos method provided by the Arpack package.\n\n\n\n\n\n","category":"method"},{"location":"man/EDCore/#LinearAlgebra.eigen-Tuple{EDMatrix}","page":"Core of Exact Diagonalization","title":"LinearAlgebra.eigen","text":"eigen(m::EDMatrix; nev=6, which=:SR, tol=0.0, maxiter=300, sigma=nothing, v₀=dtype(m)[]) -> Eigen\n\nSolve the eigen problem by the restarted Lanczos method provided by the Arpack package.\n\n\n\n\n\n","category":"method"},{"location":"man/EDCore/#LinearAlgebra.eigen-Tuple{QuantumLattices.QuantumOperators.OperatorSum{<:EDMatrix}}","page":"Core of Exact Diagonalization","title":"LinearAlgebra.eigen","text":"eigen(ms::OperatorSum{<:EDMatrix}; nev::Int=1, tol::Real=0.0, maxiter::Int=300, v₀::Union{AbstractVector, Dict{<:Sector, <:AbstractVector}, Dict{<:AbelianNumber, <:AbstractVector}}=dtype(eltype(ms))[], timer::TimerOutput=edtimer)\n\nSolve the eigen problem by the restarted Lanczos method provided by the Arpack package.\n\n\n\n\n\n","category":"method"},{"location":"man/EDCore/#QuantumLattices.:⊕-Tuple{Sector, Vararg{Union{Sector, TargetSpace}}}","page":"Core of Exact Diagonalization","title":"QuantumLattices.:⊕","text":"⊕(sector::Sector, sectors::Union{Sector, TargetSpace}...) -> TargetSpace\n⊕(target::TargetSpace, sectors::Union{Sector, TargetSpace}...) -> TargetSpace\n\nGet the direct sum of sectors and target spaces.\n\n\n\n\n\n","category":"method"},{"location":"man/EDCore/#QuantumLattices.QuantumOperators.matrix","page":"Core of Exact Diagonalization","title":"QuantumLattices.QuantumOperators.matrix","text":"matrix(ops::Operators, braket::NTuple{2, Sector}, table, dtype=valtype(eltype(ops))) -> SparseMatrixCSC{dtype, Int}\n\nGet the CSC-formed sparse matrix representation of a set of operators.\n\nHere, table specifies the order of the operator ids.\n\n\n\n\n\n","category":"function"},{"location":"man/EDCore/#QuantumLattices.QuantumOperators.matrix-Tuple{ED}","page":"Core of Exact Diagonalization","title":"QuantumLattices.QuantumOperators.matrix","text":"matrix(ed::ED, sectors::Union{AbelianNumber, Sector}...; timer::TimerOutput=edtimer, kwargs...) -> OperatorSum{<:EDMatrix}\nmatrix(ed::Algorithm{<:ED}, sectors::Union{AbelianNumber, Sector}...; kwargs...) -> OperatorSum{<:EDMatrix}\n\nGet the sparse matrix representation of a quantum lattice system in the target space.\n\n\n\n\n\n","category":"method"},{"location":"man/GreenFunctions/","page":"Green Functions","title":"Green Functions","text":"CurrentModule = ExactDiagonalization.GreenFunctions","category":"page"},{"location":"man/GreenFunctions/#Green-Functions","page":"Green Functions","title":"Green Functions","text":"","category":"section"},{"location":"man/GreenFunctions/","page":"Green Functions","title":"Green Functions","text":"Modules = [GreenFunctions]\nOrder = [:module, :constant, :type, :macro, :function]","category":"page"},{"location":"#ExactDiagonalization","page":"Home","title":"ExactDiagonalization","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: CI) (Image: codecov) (Image: ) (Image: ) (Image: 996.icu) (Image: LICENSE) (Image: LICENSE) (Image: Code Style: Blue) (Image: ColPrac: Contributor's Guide on Collaborative Practices for Community Packages)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Julia package for the exact diagonalization method in condensed matter physics.","category":"page"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Based on the symbolic operator representation of a quantum lattice system in condensed matter physics that is generated by the package QuantumLattices, exact diagonalization method is implemented for fermionic, hard-core-bosonic and spin systems.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In Julia v1.8+, please type ] in the REPL to use the package mode, then type this command:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add ExactDiagonalization","category":"page"},{"location":"#Getting-Started","page":"Home","title":"Getting Started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Examples of exact diagonalization method for quantum lattice system","category":"page"},{"location":"#Note","page":"Home","title":"Note","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Due to the fast development of this package, releases with different minor version numbers are not guaranteed to be compatible with previous ones before the release of v1.0.0. Comments are welcomed in the issues.","category":"page"},{"location":"#Contact","page":"Home","title":"Contact","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"waltergu1989@gmail.com","category":"page"},{"location":"#Python-counterpart","page":"Home","title":"Python counterpart","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"HamiltonianPy: in fact, the authors of this Julia package worked on the python package at first and only turned to Julia later.","category":"page"},{"location":"examples/HubbardModel/","page":"Fermi Hubbard Model on square lattice","title":"Fermi Hubbard Model on square lattice","text":"CurrentModule = ExactDiagonalization","category":"page"},{"location":"examples/HubbardModel/#Fermi-Hubbard-Model-on-square-lattice","page":"Fermi Hubbard Model on square lattice","title":"Fermi Hubbard Model on square lattice","text":"","category":"section"},{"location":"examples/HubbardModel/#Ground-state-energy","page":"Fermi Hubbard Model on square lattice","title":"Ground state energy","text":"","category":"section"},{"location":"examples/HubbardModel/","page":"Fermi Hubbard Model on square lattice","title":"Fermi Hubbard Model on square lattice","text":"The following codes could compute the ground state energy of the Fermi Hubbard model on square lattice.","category":"page"},{"location":"examples/HubbardModel/","page":"Fermi Hubbard Model on square lattice","title":"Fermi Hubbard Model on square lattice","text":"using QuantumLattices\nusing ExactDiagonalization\nusing LinearAlgebra: eigen\n\n# define the unitcell of the square lattice\nunitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])\n\n# define a finite 3×4 cluster of the square lattice with open boundary condition\nlattice = Lattice(unitcell, (3, 4))\n\n# define the Hilbert space (single-orbital spin-1/2 complex fermion)\nhilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(lattice))\n\n# define the quantum number of the sub-Hilbert space in which the computation to be carried out\n# here the particle number is set to be `length(lattice)` and Sz is set to be 0\nquantumnumber = SpinfulParticle(length(lattice), 0)\n\n# define the terms, i.e. the nearest-neighbor hopping and the Hubbard interaction\nt = Hopping(:t, -1.0, 1)\nU = Hubbard(:U, 8.0)\n\n# define the exact diagonalization algorithm for the Fermi Hubbard model\ned = ED(lattice, hilbert, (t, U), quantumnumber)\n\n# find the ground state and its energy\neigensystem = eigen(ed; nev=1)\n\n# Ground state energy should be -4.913259209075605\nprint(eigensystem.values)","category":"page"}]
}
