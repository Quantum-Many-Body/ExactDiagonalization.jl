module BandLanczos

using KrylovKit: Block, KrylovFactorization, KrylovIterator, OrthonormalBasis, add!!, apply, block_inner, block_qr!, block_reorthogonalize!
import KrylovKit: basis, expand!, initialize, normres, rayleighextension, rayleighquotient, residual
using LinearAlgebra: I, norm

export BandLanczosFactorization, BandLanczosIterator
export basis, expand!, initialize, normres, rayleighextension, rayleighquotient, residual

"""
    BandLanczosFactorization{T, S<:Number, SR<:Real} <: KrylovFactorization{T, S}

Band Lanczos factorization, same to [`KrylovKit.BlockLanczosFactorization`](https://jutho.github.io/KrylovKit.jl/stable/man/implementation/#KrylovKit.BlockLanczosFactorization) except that the Krylov basis vectors can be chosen to be kept or not.
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
    BandLanczosIterator{F, T, S<:Real} <: KrylovIterator{F, T}

Band Lanczos iterator.
"""
struct BandLanczosIterator{F, T, S<:Real} <: KrylovIterator{F, T}
    operator::F
    x₀::Block{T}
    maxdim::Int
    tol::S
    keepvecs::Bool
    BandLanczosIterator{F, T, S}(operator::F, x₀::Block{T}, maxdim::Int, tol::S, keepvecs::Bool) where {F, T, S<:Real} = new{F, T, S}(operator, x₀, maxdim, tol, keepvecs)
end

"""
    BandLanczosIterator(operator, x₀::Block, maxdim::Int, tol::Real=1e-12; keepvecs::Bool=true)

Construct a `BandLanczosIterator`.
"""
function BandLanczosIterator(operator, x₀::Block{T}, maxdim::Int, tol::Real=1e-12; keepvecs::Bool=true) where T
    norm(x₀)<tol && @error "BandLanczosIterator error: initial vector should not have norm zero"
    return BandLanczosIterator{typeof(operator), T, typeof(tol)}(operator, x₀, maxdim, tol, keepvecs)
end

"""
    initialize(iter::BandLanczosIterator) -> BandLanczosFactorization

Initialize a `BandLanczosFactorization` by a `BandLanczosIterator`.
"""
function initialize(iter::BandLanczosIterator)
    _, indexes = block_qr!(iter.x₀, iter.tol)
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

Expand an `BandLanczosFactorization`.
"""
function expand!(iter::BandLanczosIterator, state::BandLanczosFactorization)
    R = residual(state)
    B, indexes = block_qr!(R, iter.tol)
    k = length(state)
    bs = state.kᵣ
    bsₙ = length(indexes)
    state.H[(k+1):(k+bsₙ), (k-bs+1):k] = view(B, 1:bsₙ, 1:bs)
    state.H[(k-bs+1):k, (k+1):(k+bsₙ)] = view(B, 1:bsₙ, 1:bs)'
    push!(state.V, R[indexes])
    Rₙ, M = band_lanczosrecurrence(iter.operator, state.V, B)
    state.H[(k+1):(k+bsₙ), (k+1):(k+bsₙ)] = view(M, 1:bsₙ, 1:bsₙ)
    state.R.vec[1:bsₙ] = Rₙ.vec
    state.normᵣ = norm(Rₙ)
    state.k += bsₙ
    state.kᵣ = bsₙ
    iter.keepvecs || for _ = 1:bs popfirst!(state.V) end
    return state
end
function band_lanczosrecurrence(operator, V::OrthonormalBasis, B::AbstractMatrix)
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
    block_reorthogonalize!(AX, V)
    return AX, M
end

end