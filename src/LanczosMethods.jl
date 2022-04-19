module LanczosMethods

using LinearAlgebra: norm, dot, Hermitian

import QuantumLattices: matrix
import LinearAlgebra: eigvals

"""
    icgs(v::AbstractArray{<:Number}, bases::AbstractVector{<:AbstractVector{<:Number}}; maxiter::Int=3) -> Tuple{typeof(v), Number}

Iterative Classical Gram-Schmidt Algorithm.
"""
function icgs(v::AbstractArray{<:Number}, bases::AbstractVector{<:AbstractVector{<:Number}}; maxiter::Int=3)
    α = 0.5
    n₀ = norm(v)
    n₁ = n₀
    result = copy(v)
    for it = 1:maxiter
        for basis in bases
            overlap = dot(basis, result)
            for i = 1:length(result)
                result[i] = result[i] - overlap*basis[i]
            end
        end
        n₁ = norm(result)
        n₁>α*n₀ && break
        n₀ = n₁
    end
    return result, n₁
end

mutable struct Lanczos{T, M<:AbstractMatrix{T}, V<:AbstractVector}
    matrix::M
    initials::Vector{V}
    vectors::Vector{V}
    M::Matrix{T}
    P::Matrix{T}
    niter::Int
end
@inline Lanczos(matrix::AbstractMatrix; maxiter::Int=200) = Lanczos(matrix, rand(eltype(matrix), last(size(matrix))); maxiter=maxiter)
@inline Lanczos(matrix::AbstractMatrix, v₀::AbstractVector{<:Number}; maxiter::Int=200) = Lanczos(matrix, [v₀]; maxiter=maxiter)
function Lanczos(matrix::AbstractMatrix, initials::AbstractVector{<:AbstractVector{<:Number}}; maxiter::Int=200)
    M = zeros(eltype(matrix), maxiter, maxiter)
    P = zeros(eltype(matrix), maxiter, length(initials))
    return Lanczos(matrix, initials, eltype(initials)[], M, P, 0)
end
function Base.iterate(lanczos::Lanczos)
    lanczos.niter += 1
    candidates = copy(lanczos.initials)
    current = candidates[1]/norm(candidates[1])
    push!(lanczos.vectors, current)
    next = lanczos.matrix*current
    push!(deleteat!(candidates, 1), next)
    lanczos.M[1, 1] = dot(current, next)
    for (i, vector) in enumerate(lanczos.initials)
        lanczos.P[1, i] = dot(current, vector)
    end
    return lanczos, candidates
end
function Base.iterate(lanczos::Lanczos, candidates)
    lanczos.niter==size(lanczos.M)[1] && return nothing
    normalization, index = nothing, 1
    for candidate in candidates
        normalization = icgs(candidate, lanczos.vectors)
        normalization[2]>10^-10 && break
        index = index + 1
    end
    index>length(candidates) && return nothing
    lanczos.niter += 1
    current = normalization[1]/normalization[2]
    lanczos.niter>2*length(lanczos.initials) && deleteat!(lanczos.vectors, 1)
    push!(lanczos.vectors, current)
    next = lanczos.matrix*current
    push!(deleteat!(candidates, 1:index), next)
    for (i, vector) in enumerate(Iterators.reverse(lanczos.vectors))
        index = lanczos.niter - i + 1
        lanczos.M[index, lanczos.niter] = dot(vector, next)
        index>1 && (lanczos.M[lanczos.niter, index] = lanczos.M[index, lanczos.niter]')
    end
    for (i, vector) in enumerate(lanczos.initials)
        lanczos.P[lanczos.niter, i] = dot(current, vector)
    end
    return lanczos, candidates
end

@inline function matrix(lanczos::Lanczos, mode::Symbol=:matrix)
    mode==:matrix && return lanczos.M[1:lanczos.niter, 1:lanczos.niter]
    mode==:vectors && return lanczos.P[1:lanczos.niter, :]
    error("matrix error: mode must be :matrix or :vectors.")
end

function eigvals(lanczos::Lanczos; nev=1, tol=10^-12)
    diff = 1.0
    eigensystem = nothing
    for _ in lanczos
        new = eigvals(Hermitian(matrix(lanczos)))
        lanczos.niter>1 && (diff = norm(eigensystem[1:min(nev, lanczos.niter-1)]-new[1:min(nev, lanczos.niter-1)]))
        eigensystem = new
        diff<tol && lanczos.niter>=nev && break
    end
    lanczos.niter<nev && error("eigen error: too large nev (max=$(lanczos.niter)).")
    diff>=tol && @warn("eigen warning: maybe not converged eigenvalues (total diff=$(diff)).")
    return eigensystem[1:nev]
end

end
