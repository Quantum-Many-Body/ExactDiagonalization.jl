# Abelian bases, used for canonical spin systems by default
## Basics for Abelian bases
"""
    partition(n::Int) -> NTuple{2, Vector{Int}}

Get the default partition of n local Hilbert spaces.
"""
@inline partition(n::Int) =(cut=n÷2; (collect(1:cut), collect(cut+1:n)))

"""
    AbelianBases{A<:Abelian, N} <: Sector

A set of Abelian bases, that is, a set of bases composed from the product of local Abelian Graded spaces.

To improve the efficiency of the product of local Abelian Graded spaces, we adopt a two-step strategy:
1) partition the local spaces into several groups in each of which the local spaces are direct producted and rearranged according to the Abelian quantum numbers, and then 
2) glue the results obtained in the previous step so that a sector with a certain Abelian quantum number can be targeted.

In principle, a binary-tree strategy can be more efficient, but our two-step strategy is enough for a quantum system that can be solved by the exact diagonalization method.

The partition of the local Abelian Graded spaces is assigned by a `NTuple{N, Vector{Int}}`, with each of its element contains the sequences of the grouped local spaces specified by a table.
"""
struct AbelianBases{A<:Abelian, N} <: Sector
    quantumnumber::A
    locals::Vector{Graded{A}}
    partition::NTuple{N, Vector{Int}}
    gradeds::NTuple{N, Graded{A}}
    permutations::NTuple{N, Vector{Int}}
    record::Dict{NTuple{N, A}, Int}
    dim::Int
end
@inline id(bs::AbelianBases) = (bs.quantumnumber, bs.locals, bs.partition)
@inline dimension(bs::AbelianBases) = bs.dim
function Base.show(io::IO, bs::AbelianBases)
    @printf io "%s" "{"
    for (i, positions) in enumerate(bs.partition)
        @printf io "[%s]" join([tostr(bs.locals[position], position) for position in positions], "⊗")
        i<length(bs.partition) && @printf io "%s" " ⊗ "
    end
    @printf io ": %s}" bs.quantumnumber
end
@inline tostr(internal::Graded, order::Int) = string(internal, join('₀'+d for d in reverse(digits(order))))
function Base.match(bs₁::BS, bs₂::BS) where {BS<:AbelianBases}
    bs₁.locals==bs₂.locals || return false
    for (positions₁, positions₂) in zip(bs₁.partition, bs₂.partition)
        positions₁==positions₂ || return false
    end
    return true
end
@inline sumable(bs₁::BS, bs₂::BS) where {BS<:AbelianBases} = Abelian(bs₁) ≠ Abelian(bs₂)

"""
    Abelian(bs::AbelianBases)

Get the Abelian quantum number of a set of spin bases.
"""
@inline Abelian(bs::AbelianBases) = bs.quantumnumber

"""
    range(bs::AbelianBases) -> AbstractVector{Int}

Get the range of the target sector of an `AbelianBases` in the direct producted bases.
"""
@inline Base.range(bs::AbelianBases{ℤ₁}) = 1:dimension(bs)
function Base.range(bs::AbelianBases)
    result = Int[]
    dims = reverse(map(dimension, bs.gradeds))
    cartesian, linear = CartesianIndices(dims), LinearIndices(dims)
    total, slice = decompose(⊗(bs.gradeds...))
    for i in slice[range(total, bs.quantumnumber)]
        push!(result, linear[map(getindex, reverse(bs.permutations), cartesian[i].I)...])
    end
    return result
end

"""
    AbelianBases(locals::AbstractVector{Int}, partition::NTuple{N, AbstractVector{Int}}=partition(length(locals))) where N

Construct a set of spin bases that subjects to no quantum number conservation.
"""
@inline function AbelianBases(locals::AbstractVector{Int}, partition::NTuple{N, AbstractVector{Int}}=partition(length(locals))) where N
    return AbelianBases(Graded{ℤ₁}[Graded{ℤ₁}(0=>dim) for dim in locals], ℤ₁(0), partition)
end

"""
    AbelianBases(locals::Vector{Graded{A}}, quantumnumber::A, partition::NTuple{N, AbstractVector{Int}}=partition(length(locals))) where {N, A<:Abelian}

Construct a set of spin bases that preserves a certain symmetry specified by the corresponding Abelian quantum number.
"""
function AbelianBases(locals::Vector{Graded{A}}, quantumnumber::A, partition::NTuple{N, AbstractVector{Int}}=partition(length(locals))) where {N, A<:Abelian}
    gradeds, permutations, total, records = intermediate(locals, partition, quantumnumber)
    return AbelianBases{A, N}(quantumnumber, locals, partition, gradeds, permutations, records[1], dimension(total, quantumnumber))
end
function intermediate(spins::Vector{Graded{A}}, partition::NTuple{N, AbstractVector{Int}}, quantumnumbers::A...) where {A<:Abelian, N}
    gradeds, permutations = Graded{A}[], Vector{Int}[]
    for positions in partition
        if length(positions)>0
            graded, permutation = decompose(⊗([spins[position] for position in positions]...))
            push!(gradeds, graded)
            push!(permutations, permutation)
        end
    end
    graded = ⊗(NTuple{N, Graded{A}}(gradeds)...)
    total, fusion = merge(graded)
    records = map(quantumnumbers) do quantumnumber
        count = 1
        record = Dict{NTuple{N, A}, Int}()
        for qns in fusion[quantumnumber]
            record[qns] = count
            count += dimension(graded, qns)
        end
        record
    end
    return NTuple{N, Graded{A}}(gradeds), NTuple{N, Vector{Int}}(permutations), total::Graded{A}, records
end

"""
    matrix(index::OperatorIndex, graded::Graded, dtype::Type{<:Number}=ComplexF64) -> Matrix{dtype}

Get the matrix representation of an `OperatorIndex` on an Abelian graded space.
"""
@inline matrix(index::OperatorIndex, graded::Graded, dtype::Type{<:Number}=ComplexF64) = matrix(InternalIndex(index), graded, dtype)

"""
    matrix(op::Operator{V, <:OneAtLeast{OperatorIndex}}, braket::NTuple{2, AbelianBases}, table::AbstractDict, dtype=V) where V -> SparseMatrixCSC{dtype, Int}

Get the CSC-formed sparse matrix representation of an operator.

Here, `table` specifies the order of the operator indexes.
"""
function matrix(op::Operator{V, <:OneAtLeast{OperatorIndex}}, braket::NTuple{2, AbelianBases{ℤ₁}}, table::AbstractDict, dtype=V) where V
    bra, ket = braket
    @assert match(bra, ket) "matrix error: mismatched bra and ket."
    ms = matrices(op, ket.locals, table, dtype)
    intermediate = eltype(ms)[]
    for positions in ket.partition
        for (i, position) in enumerate(positions)
            if i==1
                push!(intermediate, ms[position])
            else
                intermediate[end] = kron(intermediate[end], ms[position])
            end
        end
    end
    result = intermediate[1]
    for i = 2:length(intermediate)
        result = kron(result, intermediate[i])
    end
    return result
end
function matrix(op::Operator{V, <:OneAtLeast{OperatorIndex}}, braket::NTuple{2, AbelianBases}, table::AbstractDict, dtype=V) where V
    bra, ket = braket
    @assert match(bra, ket) "matrix error: mismatched bra and ket."
    ms = matrices(op, ket.locals, table, dtype)
    intermediate = map((positions, graded, permutation)->blocks(ms[positions], graded, permutation), ket.partition, ket.gradeds, ket.permutations)
    result = SparseMatrixCOO(Int[], Int[], dtype[], dimension(bra), dimension(ket))
    for (row_keys, row_start) in pairs(bra.record)
        for (col_keys, col_start) in pairs(ket.record)
            kron!(result, map((row_key, col_key, m)->m[(row_key, col_key)], row_keys, col_keys, intermediate); origin=(row_start, col_start))
        end
    end
    return SparseMatrixCSC(result)
end
function matrices(op::Operator, locals::Vector{<:Graded}, table::AbstractDict, dtype)
    result = [sparse(one(dtype)*I, dimension(internal), dimension(internal)) for internal in locals]
    for (i, index) in enumerate(op)
        position = table[index]
        i==1 && (result[position] *= op.value)
        result[position] *= sparse(matrix(index, locals[position], dtype))
    end
    return result
end
function blocks(ms::Vector{<:SparseMatrixCSC}, graded::Graded, permutation::Vector{Int})
    result = Dict{Tuple{eltype(graded), eltype(graded)}, SparseMatrixCOO{eltype(eltype(ms)), Int}}()
    m = permute!(reduce(kron, ms), permutation, permutation)
    rows, vals = rowvals(m), nonzeros(m)
    for j = 1:length(graded)
        temp = [(Int[], Int[], eltype(eltype(ms))[]) for i=1:length(graded)]
        for (k, col) in enumerate(range(graded, j))
            pos = 1
            for index in nzrange(m, col)
                row = rows[index]
                val = vals[index]
                pos = findindex(row, graded, pos)
                push!(temp[pos][1], row-cumsum(graded, pos-1))
                push!(temp[pos][2], k)
                push!(temp[pos][3], val)
            end
        end
        for (i, (is, js, vs)) in enumerate(temp)
            result[(graded[i], graded[j])] = SparseMatrixCOO(is, js, vs, dimension(graded, i), dimension(graded, j))
        end
    end
    return result
end
function Base.kron!(result::SparseMatrixCOO, ms::Tuple{Vararg{SparseMatrixCOO}}; origin::Tuple{Int, Int}=(1, 1))
    ms = reverse(ms)
    row_linear = LinearIndices(map(m->1:size(m)[1], ms))
    col_linear = LinearIndices(map(m->1:size(m)[2], ms))
    for indexes in product(map(m->1:nnz(m), ms)...)
        push!(result.is, row_linear[map((index, m)->m.is[index], indexes, ms)...] + origin[1] - 1)
        push!(result.js, col_linear[map((index, m)->m.js[index], indexes, ms)...] + origin[2] - 1)
        push!(result.vs, prod(map((index, m)->m.vs[index], indexes, ms)))
    end
    return result
end

## ED based on Abelian bases for canonical spin systems
"""
    Graded{ℤ₁}(spin::Spin)
    Graded{𝕊ᶻ}(spin::Spin)

Decompose a local spin space into an Abelian graded space that preserves 1) no symmetry, and 2) spin-z component symmetry.
"""
@inline Graded{ℤ₁}(spin::Spin) = Graded{ℤ₁}(0=>Int(2*totalspin(spin)+1))
@inline Graded{𝕊ᶻ}(spin::Spin) = Graded{𝕊ᶻ}(sz=>1 for sz in -totalspin(spin):1:totalspin(spin))'

"""
    matrix(index::SpinIndex, graded::Graded, dtype::Type{<:Number}=ComplexF64) -> Matrix{dtype}

Get the matrix representation of a `SpinIndex` on an Abelian graded space.
"""
@inline function matrix(index::SpinIndex, graded::Graded{ℤ₁}, dtype::Type{<:Number}=ComplexF64)
    @assert Int(2*totalspin(index)+1)==dimension(graded) "matrix error: mismatched spin index and Abelian graded space."
    return matrix(index, dtype)
end
@inline function matrix(index::SpinIndex, graded::Graded{𝕊ᶻ}, dtype::Type{<:Number}=ComplexF64)
    S = totalspin(index)
    @assert Int(2S+1)==dimension(graded)==length(graded) && first(graded)==𝕊ᶻ(S) && last(graded)==𝕊ᶻ(-S) "matrix error: mismatched spin index and Abelian graded space."
    return matrix(index, dtype)
end

"""
    EDKind(::Type{<:SpinIndex})

Kind of the exact diagonalization method applied to a canonical quantum spin lattice system.
"""
@inline EDKind(::Type{<:SpinIndex}) = EDKind(:Abelian)

"""
    Metric(::EDKind{:Abelian}, ::Hilbert{<:Spin}) -> OperatorIndexToTuple

Get the index-to-tuple metric for a canonical quantum spin lattice system.
"""
@inline @generated Metric(::EDKind{:Abelian}, ::Hilbert{<:Spin}) = OperatorIndexToTuple(:site)

"""
    Sector(hilbert::Hilbert{<:Spin}, partition::Tuple{Vararg{AbstractVector{Int}}}=partition(length(hilbert))) -> AbelianBases
    Sector(
        quantumnumber::Abelian, hilbert::Hilbert{<:Spin}, partition::Tuple{Vararg{AbstractVector{Int}}}=partition(length(hilbert));
        table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))
    ) -> AbelianBases

Construct the Abelian bases of a spin Hilbert space with the specified quantum number.
"""
@inline function Sector(hilbert::Hilbert{<:Spin}, partition::Tuple{Vararg{AbstractVector{Int}}}=partition(length(hilbert)))
    return Sector(ℤ₁(0), hilbert, partition; table=Table(hilbert, Metric(EDKind(hilbert), hilbert)))
end
@inline function Sector(
    quantumnumber::Abelian, hilbert::Hilbert{<:Spin}, partition::Tuple{Vararg{AbstractVector{Int}}}=partition(length(hilbert));
    table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))
)
    @assert sort(vcat(partition...))==1:length(hilbert) "Sector error: incorrect partition."
    return AbelianBases(sorted_locals(typeof(quantumnumber), hilbert, table), quantumnumber, partition)
end
@inline function sorted_locals(::Type{A}, hilbert::Hilbert, table::AbstractDict) where {A<:Abelian}
    result = Graded{A}[Graded{A}(internal) for internal in values(hilbert)]
    sites = collect(keys(hilbert))
    perm = sortperm(sites; by=site->table[𝕊(site, :α)])
    return permute!(result, perm)
end

"""
    broadcast(
        ::Type{Sector}, quantumnumbers::OneAtLeast{Abelian}, hilbert::Hilbert{<:Spin}, partition::NTuple{N, AbstractVector{Int}}=partition(length(hilbert));
        table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))
    ) where N -> NTuple{fieldcount(typeof(quantumnumbers)), AbelianBases}

Construct a set of Abelian based based on the quantum numbers and a Hilbert space.
"""
function Base.broadcast(
    ::Type{Sector}, quantumnumbers::OneAtLeast{A}, hilbert::Hilbert{<:Spin}, partition::NTuple{N, AbstractVector{Int}}=partition(length(hilbert));
    table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))
) where {N, A<:Abelian}
    @assert sort(vcat(partition...))==1:length(hilbert) "Broadcast of Sector error: incorrect partition."
    quantumnumbers = OneOrMore(quantumnumbers)
    locals = sorted_locals(A, hilbert, table)
    gradeds, permutations, total, records = intermediate(locals, partition, quantumnumbers...)
    return map(quantumnumbers, records) do quantumnumber, record
        AbelianBases{A, N}(quantumnumber, locals, partition, gradeds, permutations, record, dimension(total, quantumnumber))
    end
end
