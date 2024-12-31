# module GreenFunctions

# using KrylovKit
# using LinearAlgebra
# using QuantumLattices: matrix, OperatorSum, Operators, CompositeIndex, Index, OperatorGenerator, QuantumOperator, Table, expand, FID
# using ..EDCore: Sector, EDKind
# using ..CanonicalFockSystems: BinaryBases, ⊗

# export Block, Partition, BlockVals, EDSolver, BlockGreenFunction, ClusterGreenFunction

# """
#     Block{T<:AbstractVector{Int}, O<:QuantumOperator, S<:Sector}

# The information (sign, indexes, initial operators, project operators and sector) needed in calculating a block of the Green function.
# """
# struct Block{T<:AbstractVector{Int}, O<:QuantumOperator, S<:Sector}
#     sign::Int
#     indexes::NTuple{2, T}
#     iops::Vector{O}
#     pops::Vector{O}
#     sector::S
# end

# const FockOperatorUnit = Union{FID, Index{Int, <:FID}, CompositeIndex{<:Index{Int, <:FID}}}

# function BinaryBases(bases::BinaryBases, operator::Operator{<:Number, <:Tuple{FockOperatorUnit}}, table)
#     δn = iscreation(operator[1]) ? +1 : isannihilation(operator[1]) ? -1 : error("BinaryBases error: not a creation/annihilation operator.")
#     position = table[operator[1]]
    
# end

# function isintrablock(bases::BinaryBases, operators::Vector{<:Operator{<:Number, <:Tuple{FockOperatorUnit}}})
#     @assert all(iscreation(op) for op in iops) || all(isannihilation(op) for op in iops) "Block error: iops must be all creation operators or all annihilation operators."
# end


# """
#     Block(indexes::NTuple{2, AbstractVector{Int}}, iops::AbstractVector{<:Operator}, pops::AbstractVector{<:Operator}, bs::BinaryBases)

# Construct Block with particle number/particle number and spin z component preserved bases or no particle number and no sin z component preserved bases.
# """
# function Block(indexes::NTuple{2, AbstractVector{Int}}, iops::AbstractVector{<:Operator}, pops::AbstractVector{<:Operator}, bs::BinaryBases)
#     @assert all(iscreation(op) for op in iops) || all(isannihilation(op) for op in iops) "Block error: iops must be all creation operators or all annihilation operators."

#     (arr, _), ps = indexes, [bs.quantumnumbers[i].N for i in eachindex(bs.quantumnumbers)]
#     id = [findall(bit -> isone(bit), [isone(b, bit) for bit in 1:64]) for b in bs.stategroups]
#     nambu = [op.id[1].index.iid.nambu for op in iops[1]][1]
#     isnothing(findfirst(x -> x == arr, id)) ? index=1 : index=findfirst(x -> x == arr, id)
#     nambu == 1 ? ps[index] -= 1 : ps[index] += 1
#     isnan(ps[1]) ? nbs=bs : nbs=reduce(⊗, [BinaryBases(id[i], convert(Int64, ps[i])) for i in eachindex(id)])
#     return Block(nambu, indexes, iops, pops, nbs)
# end

# """
#     Block(spindwups::AbstractVector, indexes::AbstractVector, iops::AbstractVector, pops::AbstractVector, bs::BinaryBases)

# Construct Block with only spin z component perserved bases
# """
# function Block(spindwups::AbstractVector, indexes::AbstractVector, iops::AbstractVector, pops::AbstractVector, bs::BinaryBases) where M
#     spin, sz, nambu = [op.id[1].index.iid.spin for op in iops[1]][1], bs.quantumnumbers[1].Sz, [op.id[1].index.iid.nambu for op in iops[1]][1]
#     nambu == 1 ? sz -= spin : sz += spin
#     nbs = BinaryBases(spindwups..., convert(Float64,sz))
#     return Block(nambu, indexes, iops, pops, nbs)
# end

# """
#     Partition{S<:Symbol, B<:Block, L<:AbstractVector{B}, R<:Sector}
#     Partition(::Val{:N}, table::Table, bs::BinaryBases)
#     Partition(::Val{:S}, table::Table, bs::BinaryBases)
#     Partition(::Val{:A}, table::Table, bs::BinaryBases)
#     Partition(::Val{:F}, table::Table, bs::BinaryBases)

# Divide Green's function into different blocks according to the different terms contained in the Hamiltonian.
#     The Hamiltonian of the system has:
#     (1) only normal terms -> :N
#     (2) spin-filp terms -> :S
#     (3) anomalous terms -> :A
#     (4) all the above -> :F
# """
# struct Partition{B<:Block, R<:Sector}
#     mode::Symbol
#     lesser::Vector{B}
#     greater::Vector{B}
# end

# @inline function Partition(
#     mode::Symbol, lattice::AbstractLattice, hilbert::Hilbert, quantumnumber::AbelianNumber, basistype=UInt;
#     table=Table(hilbert, Metric(EDKind(hilbert), hilbert))
# )
#     return Partition(Val(mode), hilbert, quantumnumber, basistype; table=table)
# end

# function Partition(
#     ::Val{:N}, lattice::AbstractLattice, hilbert::Hilbert, quantumnumber::AbelianNumber, basistype=UInt;
#     table=Table(hilbert, Metric(EDKind(hilbert), hilbert))
# )
    

# end






# Partition(sym::Symbol, table::Table, bs::BinaryBases) = Partition(Val(sym), table, bs)
# function Partition(::Val{:N}, table::Table, bs::BinaryBases)
#     seqs = [(seq..., i) for seq in sort(collect(keys(table)), by = x -> table[x]), i in 1:2]
#     id₁, id₂ = seqs[:,1], seqs[:,2]
#     ops₁, ops₂ = [[Operators(1*CompositeIndex(Index(key[2], FID{:f}(key[3], key[1], key[4])), [0.0, 0.0], [0.0, 0.0])) for key in id] for id in [id₁, id₂]]
#     arrs = [Vector(1:length(id₁))[(i-1)*(length(Vector(1:length(id₁)))÷(length(bs.stategroups)))+1:i*(length(Vector(1:length(id₁)))÷(length(bs.stategroups)))] for i in 1:length(bs.stategroups)]
#     lesser, greater = [Block([arr, arr], ops₁[arr], ops₁[arr], bs) for arr in arrs], [Block([arr, arr], ops₂[arr],ops₂[arr], bs) for arr in arrs]
#     return Partition(:N, lesser, greater, bs)
# end
# function Partition(::Val{:S}, table::Table, bs::BinaryBases)
#     seqs = [(seq..., i) for seq in sort(collect(keys(table)), by = x -> table[x]), i in 1:2]
#     id₁, id₂ = seqs[:,1], seqs[:,2]
#     ops₁, ops₂ = [[Operators(1*CompositeIndex(Index(key[2], FID{:f}(key[3], key[1], key[4])), [0.0, 0.0], [0.0, 0.0])) for key in id] for id in [id₁, id₂]]
#     arr = Vector(1:length(id₁))
#     lesser, greater = [Block([arr,arr], ops₁[arr], ops₁[arr], bs)], [Block([arr,arr], ops₂[arr], ops₂[arr], bs)]
#     return Partition(:S, lesser, greater, bs)
# end
# function Partition(::Val{:A}, table::Table, bs::BinaryBases)
#     seqs = [(seq..., i) for seq in sort(collect(keys(table)), by = x -> table[x]), i in 1:2]
#     id₁, id₂ = [seqs[:,1]...,seqs[:,2]...], [seqs[:,2]...,seqs[:,1]...]
#     ops₁, ops₂ = [[Operators(1*CompositeIndex(Index(key[2], FID{:f}(key[3], key[1], key[4])), [0.0, 0.0], [0.0, 0.0])) for key in id] for id in [id₁, id₂]]
#     ns = 2*abs(id₁[1][1])+1
#     arrs = [Vector(1:length(id₁))[(i-1)*(length(Vector(1:length(id₁)))÷(2*(ns)))+1:i*(length(Vector(1:length(id₁)))÷(2*(ns)))] for i in 1:convert(Int, 2*(ns))]
#     brrs = vcat(arrs[1:div(length(arrs), 2)], reverse(arrs[1:div(length(arrs), 2)]))
#     lesser = [Block(arrs[1:div(length(arrs), 2)], [brrs[i], arrs[i]], ops₁[brrs[i]], ops₁[arrs[i]], bs) for i in eachindex(arrs)]
#     greater = [Block(arrs[1:div(length(arrs), 2)], [arrs[i], brrs[i]], ops₂[brrs[i]], ops₂[arrs[i]], bs) for i in eachindex(arrs)]
#     return Partition(:A, lesser, greater, bs)
# end
# function Partition(::Val{:F}, table::Table, bs::BinaryBases)
#     seqs = [(seq..., i) for seq in sort(collect(keys(table)), by = x -> table[x]), i in 1:2]
#     id₁, id₂ = [seqs[:,1]...,seqs[:,2]...], [seqs[:,2]...,seqs[:,1]...]
#     ops₁, ops₂ = [[Operators(1*CompositeIndex(Index(key[2], FID{:f}(key[3], key[1], key[4])), [0.0, 0.0], [0.0, 0.0])) for key in id] for id in [id₁, id₂]]
#     arr, brr = 1:(length(id₁)÷2), (length(id₁)÷2+1):length(id₁)
#     lesser, greater = [Block([arr,arr], ops₁[arr], ops₁[arr], bs), Block([arr,brr], ops₁[arr], ops₁[brr], bs)], [Block([arr,arr], ops₂[arr], ops₂[arr], bs), Block([brr,arr], ops₂[arr], ops₂[brr], bs)]
#     return Partition(:F, lesser, greater, bs)
# end

# """
#     genkrylov(matrix::AbstractMatrix, istate::AbstractVector, m::Int)

# Generate a krylov subspace with Lanczos iteration method.
# """
# function genkrylov(matrix::AbstractMatrix, istate::AbstractVector, m::Int)
#     orth = KrylovKit.ModifiedGramSchmidt()
#     iterator = LanczosIterator(matrix, istate, orth)
#     factorization = KrylovKit.initialize(iterator)
#     for _ in 1:m-1
#         KrylovKit.expand!(iterator, factorization)
#     end
#     basisvectors = basis(factorization)
#     T = rayleighquotient(factorization)
#     a, c = [0.0; T.ev[1:end-1]], [T.ev[1:end-1]; 0.0]
#     return basisvectors, a, T.dv, c
# end 

# """
#     mutable struct BlockVals{I<:Int, D<:AbstractVector, R<:AbstractVector, N<:AbstractVector, P<:AbstractMatrix}

# The Lanczos' information needed in calculating the cluster Green function.
# """
# mutable struct BlockVals{I<:Int, D<:AbstractVector, R<:AbstractVector, N<:AbstractVector, P<:AbstractMatrix}
#     sign::I
#     const block::D
#     const abc::R
#     const norms::N
#     const projects::P
# end
# Base.copy(bv::BlockVals) = BlockVals(bv.sign, bv.block, bv.abc, bv.norms, bv.projects)

# """
#     BlockVals(block::Block, gs::AbstractVector, rops::OperatorSum, bs::BinaryBases, table::Table; m::Int=200)

# Obtain the Lanczos' information needed in calculating the cluster Green function.
# """
# function BlockVals(block::Block, gs::AbstractVector, rops::OperatorSum, bs::BinaryBases, table::Table; m::Int=200)
#     H = matrix(rops, (block.sector, block.sector), table)
#     istates, pstates = [(matrix(op, (block.sector, bs), table)*gs)[:,1] for op in block.iops], [(matrix(op, (block.sector, bs), table)*gs)[:,1] for op in block.pops]
#     abc, norms, projects = kryvals(H, istates, pstates, m)
#     return BlockVals(block.sign, block.indexes, abc, norms, projects)
# end
# function kryvals(H::AbstractMatrix, istates::AbstractVector, pstates::AbstractVector, m::Int=200)
#     avs, bvs, cvs, norms, projects = Vector{Vector{Float64}}(undef,length(istates)), Vector{Vector{Float64}}(undef,length(istates)), Vector{Vector{Float64}}(undef,length(istates)), Vector{Float64}(undef,length(istates)), Matrix{Vector{ComplexF64}}(undef,length(istates),length(istates))
#     for i in eachindex(istates)
#         krybasis, avs[i], bvs[i], cvs[i] = genkrylov(H, istates[i], m)  
#         norms[i] = norm(istates[i])
#         for j in eachindex(pstates)
#             projects[i, j] = KrylovKit.project!!(zeros(ComplexF64, m), krybasis, pstates[j])
#         end
#     end
#     return ([avs, bvs, cvs], norms, projects)
# end

# """
#     EDSolver{R<:Real, B<:BlockVals, S<:AbstractVector{B}, I<:Integer}

# The exact diagonalization solver of a certain system.
# """
# struct EDSolver{E<:ED, B<:BlockVals}
#     ed::E
#     gse::Float64
#     lvals::Vector{B}
#     gvals::Vector{B}
#     lens::Int
# end

# """
#     EDSolver(::EDKind{:FED}, sym::Symbol, refergenerator::OperatorGenerator, bs::BinaryBases, table::Table; m::Int=200)

# Construct the exact diagonalization solver of a certain system.
# """
# function EDSolver(::EDKind{:FED}, parts::Partition, refergenerator::OperatorGenerator, bs::BinaryBases, table::Table; m::Int=200)
#     rops = expand(refergenerator)
#     Hₘ = matrix(rops, (bs, bs), table)
#     vals, vecs, _  = KrylovKit.eigsolve(Hₘ, 1, :SR, Float64)
#     gse, gs = real(vals[1]), vecs[1]
#     lesser, greater = parts.lesser, parts.greater
#     lvals, gvals = [BlockVals(bl, gs, rops, bs, table; m=m) for bl in lesser], [BlockVals(bg, gs, rops, bs, table; m=m) for bg in greater]
#     lens = maximum([maximum([maximum([maximum([maximum(arr) for arr in block]) for block in val.block]) for val in vals]) for vals in [lvals, gvals]])
#     return EDSolver(gse, lvals, gvals, lens)
# end

# """
#     BlockGreenFunction(gse::Real, blockvals::BlockVals, ω::Complex)

# Calculate a block of cluster Green function.
# """
# function BlockGreenFunction(gse::Real, blockvals::BlockVals, ω::Complex)
#     (avs, bvs, cvs),  norms, proj = blockvals.abc, blockvals.norms, blockvals.projects
#     bgfm, d = zeros(ComplexF64, length(norms), length(norms)), [1.0;zeros(length(avs[1])-1)]
#     blockvals.sign==1 ? bv=[(ω - gse) .+ bvs[i] for i in 1:length(norms)] : bv=[(ω + gse) .- bvs[i] for i in 1:length(norms)]
#     blockvals.sign==1 ? tmpv=[thomas(avs[i],bv[i],cvs[i],d,length(avs[1])) for i in 1:length(norms)] : tmpv=[thomas(-avs[i],bv[i],-cvs[i],d,length(avs[1])) for i in 1:length(norms)]
#     for i in eachindex(norms), j in eachindex(norms)
#             bgfm[i, j] = dot(proj[i, j], tmpv[i])*norms[i]
#     end
#     return bgfm
# end
# function thomas(a::AbstractVector, b::AbstractVector, c::AbstractVector, d::AbstractVector, n::Int)
#     x = Complex.(d)
#     cp = Complex.(c)
#     cp[1] /= b[1]
#     x[1] /= b[1]
#     for i = 2:n
#         scale = 1.0 / (b[i] - cp[i-1]*a[i])
#         cp[i] *= scale
#         x[i] = (x[i] - a[i]*x[i-1])*scale
#     end
#     for i = n-1:-1:1
#         x[i] -= (cp[i]*x[i+1])
#     end
#     return x
# end

# """
#     ClusterGreenFunction(normal::Bool, kind::Symbol, solver::EDSolver, ω::Complex)
#     ClusterNormalGreenFunction(kind::Symbol, solver::EDSolver, ω::Complex)
#     ClusterGorkovGreenFunction(kind::Symbol, solver::EDSolver, ω::Complex)

# Calculate the cluster Green function with certain frequence ω
# """
# function ClusterGreenFunction(normal::Bool, kind::Symbol, solver::EDSolver, ω::Complex)
#     (normal||length(solver.lvals)==1) ? cgf=ClusterNormalGreenFunction(kind::Symbol, solver::EDSolver, ω::Complex) : cgf=ClusterGorkovGreenFunction(kind::Symbol, solver::EDSolver, ω::Complex)
#     return cgf
# end
# function ClusterNormalGreenFunction(kind::Symbol, solver::EDSolver, ω::Complex)
#     clm, cgm = zeros(ComplexF64, solver.lens, solver.lens), zeros(ComplexF64, solver.lens, solver.lens)
#     for lval in solver.lvals
#         clm[lval.block...] = BlockGreenFunction(solver.gse, lval, ω)
#     end
#     for gval in solver.gvals
#         cgm[gval.block...] = transpose(BlockGreenFunction(solver.gse, gval, ω))
#     end
#     kind==:f ? cgf=clm+cgm : (kind==:b ? cgf=cgm-clm : cgf=zeros(ComplexF64, solver.lens, solver.lens))
#     return cgf
# end
# function ClusterGorkovGreenFunction(kind::Symbol, solver::EDSolver, ω::Complex)
#     clm, cgm = zeros(ComplexF64, solver.lens, solver.lens), zeros(ComplexF64, solver.lens, solver.lens)
#     for lval in solver.lvals
#         clm[lval.block...] = BlockGreenFunction(solver.gse, lval, ω)
#         glval = copy(lval)
#         glval.sign = 2
#         if lval.block[1] == lval.block[2]
#             arr = lval.block[1] .+ (solver.lens ÷ 2) 
#             cgm[arr, arr] = transpose(BlockGreenFunction(solver.gse, glval, ω))
#         else
#             cgm[lval.block...] = BlockGreenFunction(solver.gse, glval, ω)
#         end
#     end
#     for gval in solver.gvals
#         cgm[gval.block...] = transpose(BlockGreenFunction(solver.gse, gval, ω))
#         lgval = copy(gval)
#         lgval.sign = 1
#         if gval.block[1] == gval.block[2]
#             arr = gval.block[1] .+ (solver.lens ÷ 2)
#             clm[arr, arr] = BlockGreenFunction(solver.gse, lgval, ω)
#         else
#             clm[gval.block...] = transpose(BlockGreenFunction(solver.gse, lgval, ω))
#         end
#     end
#     cgm[Vector(1:(solver.lens÷2)), Vector(((solver.lens÷2)+1):solver.lens)] = transpose(cgm[Vector(1:(solver.lens÷2)), Vector(((solver.lens÷2)+1):solver.lens)])
#     clm[Vector(((solver.lens÷2)+1):solver.lens), Vector(1:(solver.lens÷2))] = transpose(clm[Vector(((solver.lens÷2)+1):solver.lens), Vector(1:(solver.lens÷2))])
#     kind==:f ? cgf=clm+cgm : (kind==:b ? cgf=cgm-clm : cgf=zeros(ComplexF64, solver.lens, solver.lens))
#     return cgf
# end


# end # module
