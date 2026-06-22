module BandLanczos

using KrylovKit: Block, KrylovFactorization, KrylovIterator, ModifiedGramSchmidt, ModifiedGramSchmidt2, Orthogonalizer, OrthonormalBasis
using KrylovKit: add!!, apply, block_inner, inner, orthogonalize!!, scale!!, zerovector!!
using LinearAlgebra: I, norm
import KrylovKit: basis, block_qr!, block_reorthogonalize!, expand!, initialize, normres, rayleighextension, rayleighquotient, residual

export BandLanczosFactorization, BandLanczosIterator, ModifiedGramSchmidt, ModifiedGramSchmidt2, Orthogonalizer
export basis, expand!, initialize, normres, rayleighextension, rayleighquotient, residual

"""
    mutable struct BandLanczosFactorization{T, S<:Number, SR<:Real} <: KrylovFactorization{T, S}

Band Lanczos factorization of a real symmetric or complex hermitian linear map `A`, storing a partial factorization of the form

```julia
A * V = V * H + R * B'
```

where `V` is an orthonormal basis of the Krylov subspace, `H` is a block tridiagonal Rayleigh quotient matrix, `R` is the residual block, and `B = [0; I]` contains an identity matrix in the last `kᵣ` rows.

Compared to [`KrylovKit.BlockLanczosFactorization`](https://jutho.github.io/KrylovKit.jl/stable/man/implementation/#KrylovKit.BlockLanczosFactorization), which requires [`ModifiedGramSchmidt`](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.ModifiedGramSchmidt) and always keeps the full basis, this factorization is compatible with both single-pass [`ModifiedGramSchmidt`](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.ModifiedGramSchmidt) and [`ModifiedGramSchmidt2`](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.ModifiedGramSchmidt2), and can drop old basis vectors to save memory (controlled by the iterator, not the factorization itself).

## Fields

- `k::Int`: current dimension of the Krylov subspace (number of basis vectors).
- `V::OrthonormalBasis{T}`: orthonormal basis of the Krylov subspace.
- `H::Matrix{S}`: block tridiagonal Rayleigh quotient matrix, preallocated to `maxdim × maxdim`.
- `R::Block{T}`: residual block; only the first `kᵣ` vectors are valid.
- `kᵣ::Int`: number of vectors in the current residual block.
- `normᵣ::SR`: Frobenius norm of the residual block.

## Interface

- `length(fact)` → `fact.k`
- `basis(fact)` → `fact.V` (requires `keepvecs=true`)
- `rayleighquotient(fact)` → `fact.H[1:k, 1:k]`
- `residual(fact)` → `fact.R[1:kᵣ]`
- `normres(fact)` → `fact.normᵣ`
- `rayleighextension(fact)` → `[0; I]` matrix of size `(k, kᵣ)`

See also: [`BandLanczosIterator`](@ref), [`expand!`](@ref), [`initialize`](@ref).
"""
mutable struct BandLanczosFactorization{T, S<:Number, SR<:Real} <: KrylovFactorization{T, S}
    k::Int
    V::OrthonormalBasis{T}
    H::Matrix{S}
    R::Block{T}
    kᵣ::Int
    normᵣ::SR
end
@inline Base.length(fact::BandLanczosFactorization) = fact.k
@inline basis(fact::BandLanczosFactorization) = length(fact.V)==length(fact) ? fact.V : error("basis error: not keeping vectors during band Lanczos factorization.")
@inline rayleighquotient(fact::BandLanczosFactorization) = fact.H[1:length(fact), 1:length(fact)]
@inline residual(fact::BandLanczosFactorization) = fact.R[1:(fact.kᵣ)]
@inline normres(fact::BandLanczosFactorization) = fact.normᵣ
@inline rayleighextension(fact::BandLanczosFactorization) = vcat(zeros(eltype(fact.H), fact.k-fact.kᵣ, fact.kᵣ), Matrix{eltype(fact.H)}(I, fact.kᵣ, fact.kᵣ))

"""
    struct BandLanczosIterator{F, T, S<:Real, SR<:Real, OR<:Orthogonalizer, OO<:Orthogonalizer} <: KrylovIterator{F, T}

Iterator that produces a progressively expanding [`BandLanczosFactorization`](@ref) of a real symmetric or complex hermitian linear map `f` with a starting block `x₀`.

Compared to [`KrylovKit.BlockLanczosIterator`](https://jutho.github.io/KrylovKit.jl/stable/man/implementation/#KrylovKit.BlockLanczosIterator), which only supports [`ModifiedGramSchmidt`](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.ModifiedGramSchmidt) and always keeps the full Krylov basis, this iterator:

- Supports both single-pass [`ModifiedGramSchmidt`](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.ModifiedGramSchmidt) and [`ModifiedGramSchmidt2`](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.ModifiedGramSchmidt2) for both the QR step and the reorthogonalization step, independently selectable.
- Supports optional basis truncation via `keepvecs` to save memory.
- Separates the Lanczos convergence tolerance (`tol`) from the QR rank-determination
  tolerance (`tolᵣ`), allowing fine-grained control over numerical behavior.

## Fields

- `operator::F`: the linear map (function or matrix).
- `x₀::Block{T}`: initial block of vectors.
- `maxdim::Int`: maximum dimension of the Krylov subspace.
- `tol::S`: Lanczos convergence tolerance — iteration stops when `normres < tol`.
- `tolᵣ::SR`: QR rank-determination tolerance — a vector with `norm < tolᵣ` is considered numerically zero in `block_qr!`.
- `keepvecs::Bool`: whether to retain all basis vectors (if `false`, old vectors are dropped each iteration to save memory).
- `orthᵣ::OR`: [`Orthogonalizer`](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.Orthogonalizer) used in [`block_qr!`](@ref) for rank determination of initial and residual blocks.
- `orthₒ::OO`: [`Orthogonalizer`](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.Orthogonalizer) used in [`block_reorthogonalize!`](@ref) for reorthogonalization against the full Krylov basis during the Lanczos recurrence.

See also: [`BandLanczosFactorization`](@ref), [`initialize`](@ref), [`expand!`](@ref).
"""
struct BandLanczosIterator{F, T, S<:Real, SR<:Real, OR<:Orthogonalizer, OO<:Orthogonalizer} <: KrylovIterator{F, T}
    operator::F
    x₀::Block{T}
    maxdim::Int
    tol::S
    tolᵣ::SR
    orthᵣ::OR
    orthₒ::OO
    keepvecs::Bool
end
function BandLanczosIterator(
    operator, x₀::Block, maxdim::Int, tol::Real=2e-8, tolᵣ::Real=tol=1e-8, orthᵣ::Orthogonalizer=ModifiedGramSchmidt2(), orthₒ::Orthogonalizer=ModifiedGramSchmidt();
    keepvecs::Bool=true
)
    norm(x₀) < tolᵣ && @error "BandLanczosIterator error: initial vector should not have norm zero"
    return BandLanczosIterator(operator, x₀, maxdim, tol, tolᵣ, orthᵣ, orthₒ, keepvecs)
end

"""
    initialize(iter::BandLanczosIterator) -> BandLanczosFactorization

Initialize the band Lanczos factorization from the starting block `iter.x₀`.

Performs a QR decomposition of the initial block via [`block_qr!`](@ref) (using`iter.tolᵣ` and `iter.orthᵣ`), applies the operator to the orthonormalized block, computes the first block of the Rayleigh quotient, and constructs the initial residual.
"""
function initialize(iter::BandLanczosIterator)
    _, indexes = block_qr!(iter.x₀, iter.tolᵣ, iter.orthᵣ)
    X = iter.x₀[indexes]
    V = OrthonormalBasis(X.vec)
    bs = length(X)
    AX = apply(iter.operator, X)
    M = block_inner(X, AX)
    H = zeros(eltype(M), iter.maxdim, iter.maxdim)
    H[1:bs, 1:bs] = view(M, 1:bs, 1:bs)
    for j in 1:length(X)
        for i in 1:length(X)
            AX[j] = add!!(AX[j], X[i], -M[i, j])
        end
    end
    normᵣ = norm(AX)
    return BandLanczosFactorization(bs, V, H, AX, bs, normᵣ)
end

"""
    expand!(iter::BandLanczosIterator, state::BandLanczosFactorization) -> BandLanczosFactorization

Expand the band Lanczos factorization by one block step.

1. QR-decompose the residual block via `block_qr!(R, iter.tolᵣ, iter.orthᵣ)`. If the residual is exhausted (`bsₙ == 0`), return the state unchanged.
2. Append the QR-orthonormalized residual to the basis.
3. Fill the off-diagonal blocks of `H` with the QR factor `B`.
4. Compute the next residual via [`band_lanczosrecurrence`](@ref) using `iter.orthₒ`.
5. Update the diagonal block of `H` with the Rayleigh quotient of the new basis vectors.
6. Update `normᵣ` and `kᵣ`.
"""
function expand!(iter::BandLanczosIterator, state::BandLanczosFactorization)
    R = residual(state)
    B, indexes = block_qr!(R, iter.tolᵣ, iter.orthᵣ)
    k = length(state)
    bs = state.kᵣ
    bsₙ = length(indexes)
    state.H[(k+1):(k+bsₙ), (k-bs+1):k] = view(B, 1:bsₙ, 1:bs)
    state.H[(k-bs+1):k, (k+1):(k+bsₙ)] = view(B, 1:bsₙ, 1:bs)'
    push!(state.V, R[indexes])
    Rₙ, M = band_lanczosrecurrence(iter.operator, state.V, B, iter.orthₒ)
    state.H[(k+1):(k+bsₙ), (k+1):(k+bsₙ)] = view(M, 1:bsₙ, 1:bsₙ)
    state.R.vec[1:bsₙ] = Rₙ.vec
    state.normᵣ = norm(Rₙ)
    state.k += bsₙ
    state.kᵣ = bsₙ
    iter.keepvecs || for _ = 1:bs popfirst!(state.V) end
    return state
end

"""
    band_lanczosrecurrence(operator, V::OrthonormalBasis, B::AbstractMatrix, orth::Orthogonalizer) -> (AX, M)

Perform one step of the band Lanczos recurrence.

Given the current orthonormal basis `V`, the QR factor `B` (of size `bs × bsₚ`), and an [`Orthogonalizer`](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.Orthogonalizer) `orth`:

1. Apply the operator to the newest block `X = V[end-bs+1:end]`.
2. Compute the Rayleigh quotient `M = X' * A * X`.
3. Subtract projections onto the current block `X` (via `M`) and the previous block `Xₚ` (via `B`) to form the next residual.
4. Reorthogonalize against the full Krylov basis via `block_reorthogonalize!` using `orth`.

Returns the new residual block `AX` and the Rayleigh quotient block `M`.
"""
function band_lanczosrecurrence(operator, V::OrthonormalBasis, B::AbstractMatrix, orth::Orthogonalizer)
    bs, bsₚ = size(B)
    X = Block(V[(end-bs+1):end])
    AX = apply(operator, X)
    M = block_inner(X, AX)
    Xₚ = Block(V[(end-bsₚ-bs+1):(end-bs)])
    @inbounds for j in 1:length(X)
        for i in 1:length(X)
            AX[j] = add!!(AX[j], X[i], -M[i, j])
        end
        for i in 1:length(Xₚ)
            AX[j] = add!!(AX[j], Xₚ[i], -conj(B[j, i]))
        end
    end
    block_reorthogonalize!(AX, V, orth)
    return AX, M
end

"""
    block_qr!(block::Block, tol::Real, orth::Orthogonalizer) -> (R, good_idx)

QR factorization of a block of vectors using the specified orthogonalizer.

- `orth = ModifiedGramSchmidt()`: single-pass Modified Gram-Schmidt (delegates to KrylovKit's standard `block_qr!`).
- `orth = ModifiedGramSchmidt2()`: Modified Gram-Schmidt with one reorthogonalization pass. Each vector is orthogonalized against all previous vectors twice, which significantly improves numerical orthogonality.

Returns the upper-triangular factor `R` (size `r × n` where `r` is the numerical rank) and a vector `good_idx` of indices of the linearly independent columns. Vectors whose norm after orthogonalization falls below `tol` are set to zero and excluded from `good_idx`.
"""
@inline block_qr!(block::Block, tol::Real, ::ModifiedGramSchmidt) = block_qr!(block, tol)
function block_qr!(block::Block, tol::Real, ::ModifiedGramSchmidt2)
    n = length(block)
    rank_shrink = false
    idx = trues(n)
    r₁₁ = inner(block[1], block[1])
    R = zeros(typeof(r₁₁), n, n)
    β = sqrt(real(r₁₁))
    if β > tol
        R[1, 1] = β
        block[1] = scale!!(block[1], 1 / β)
    else
        block[1] = zerovector!!(block[1])
        rank_shrink = true
        idx[1] = false
    end
    @inbounds for j in 2:n
        for pass in 1:2
            for i in 1:(j - 1)
                s = inner(block[i], block[j])
                if pass == 1
                    R[i, j] = s
                else
                    R[i, j] += s
                end
                block[j] = add!!(block[j], block[i], -s)
            end
        end
        β = norm(block[j])
        if β > tol
            R[j, j] = β
            block[j] = scale!!(block[j], 1 / β)
        else
            block[j] = zerovector!!(block[j])
            rank_shrink = true
            idx[j] = false
        end
    end
    if rank_shrink
        good_idx = findall(idx)
        return R[good_idx, :], good_idx
    else
        return R, collect(Int, 1:n)
    end
end

"""
    block_reorthogonalize!(R::Block{T}, V::OrthonormalBasis{T}, orth::Orthogonalizer) where {T}

Reorthogonalize the vectors in `R` against the orthonormal basis `V` using the specified orthogonalizer.

For each vector `R[i]`, projects out its components along every direction in `V`: `R[i] = R[i] - Σⱼ ⟨R[i], V[j]⟩ V[j]`.

- `orth = ModifiedGramSchmidt()`: single-pass MGS (delegates to KrylovKit's standard `block_reorthogonalize!`).
- `orth = ModifiedGramSchmidt2()`: two-pass MGS, providing better orthogonality.

It is assumed that `V` is already orthonormal.
"""
@inline block_reorthogonalize!(R::Block{T}, V::OrthonormalBasis{T}, ::ModifiedGramSchmidt) where {T} = block_reorthogonalize!(R, V)
function block_reorthogonalize!(R::Block{T}, V::OrthonormalBasis{T}, ::ModifiedGramSchmidt2) where {T}
    for i in 1:length(R)
        for q in V
            R[i], _ = orthogonalize!!(R[i], q, ModifiedGramSchmidt2())
        end
    end
    return R
end

end