"""
    AbstractGreenFunction{T<:Number} <: Function

Abstract type for Green's functions obtained by Krylov space expansion.
"""
abstract type AbstractGreenFunction{T<:Number} <: Function end
@inline Base.:(==)(gf₁::AbstractGreenFunction, gf₂::AbstractGreenFunction) = ==(efficientoperations, gf₁, gf₂)
@inline Base.isequal(gf₁::AbstractGreenFunction, gf₂::AbstractGreenFunction) = isequal(efficientoperations, gf₁, gf₂)

"""
    eltype(gf::AbstractGreenFunction)
    eltype(::Type{<:AbstractGreenFunction{T}}) where {T<:Number}

Get the eltype of an `AbstractGreenFunction`.
"""
@inline Base.eltype(gf::AbstractGreenFunction) = eltype(typeof(gf))
@inline Base.eltype(::Type{<:AbstractGreenFunction{T}}) where {T<:Number} = T

"""
    size(gf::AbstractGreenFunction) -> NTuple{2, Int}

Get the size of an `AbstractGreenFunction`.
"""
@inline Base.size(gf::AbstractGreenFunction) = (rank(gf), rank(gf))

"""
    (gf::AbstractGreenFunction)(ω::Number; sign::Bool=false) -> Matrix{promote_type(typeof(ω), eltype(gf))}

Get the values of an `AbstractGreenFunction` at `ω`.

When `sign` is `true`, the opposite will be taken in the result.
"""
@inline (gf::AbstractGreenFunction)(ω::Number; sign::Bool=false) = gf(zeros(promote_type(typeof(ω), eltype(gf)), size(gf)), ω; sign)

"""
    GreenFunction{T<:Number, V<:Real} <: AbstractGreenFunction{T}

Green function obtained by Krylov space expansion.
"""
struct GreenFunction{T<:Number, V<:Real} <: AbstractGreenFunction{T}
    Q::Matrix{T}
    E::Vector{V}
    sign::Bool
    function GreenFunction(Q::AbstractMatrix{<:Number}, E::AbstractVector{<:Number}, sign::Bool)
        @assert size(Q, 2)==length(E) "GreenFunction error: mismatched E and Q."
        new{eltype(Q), eltype(E)}(Q, E, sign)
    end
end

"""
    rank(gf::GreenFunction) -> Int

Get the rank of a `GreenFunction`.
"""
@inline rank(gf::GreenFunction) = size(gf.Q, 1)

"""
    dimension(gf::GreenFunction) -> Int

Get the dimension of the Krylov space expanded to obtained a `GreenFunction`.
"""
@inline dimension(gf::GreenFunction) = size(gf.Q, 2)

"""
    (gf::GreenFunction)(dest::AbstractMatrix{<:Number}, ω::Number; sign::Bool=false) -> typeof(dest)

Get the values of a `GreenFunction` at `ω` and add the result to `dest`.
"""
function (gf::GreenFunction)(dest::AbstractMatrix{<:Number}, ω::Number; sign::Bool=false)
    if gf.sign
        # case of lesser Green's function
        for i = 1:rank(gf), j = 1:rank(gf)
            for k in 1:dimension(gf)
                coeff = (-1)^sign / (ω+gf.E[k])
                dest[i, j] += coeff * Q[j, k] * (Q[i, k])'
            end
        end
    else
        # case of greater Green's function
        for i = 1:rank(gf), j = 1:rank(gf)
            for k in 1:dimension(gf)
                coeff = (-1)^sign / (ω-gf.E[k])
                dest[i, j] += coeff * Q[i, k] * (Q[j, k])'
            end
        end
    end
    return dest
end

"""
    reset!(
        gf::GreenFunction, H::AbstractMatrix{<:Number}, V::AbstractVector{AbstractVector{<:Number}}, E₀::Real;
        ranks::AbstractVector{<:Integer}=1:rank(gf), dimensions::AbstractVector{<:Integer}=1:dimension(gf)
    ) -> typeof(gf)

Reset (a block) of `GreenFunction`.
"""
function reset!(
    gf::GreenFunction, H::AbstractMatrix{<:Number}, V::AbstractVector{AbstractVector{<:Number}}, E₀::Real;
    ranks::AbstractVector{<:Integer}=1:rank(gf), dimensions::AbstractVector{<:Integer}=1:dimension(gf)
)
    @assert allequal(size(H)) "reset! error: input Hamiltonian ($(join(size(H), "×"))) is not a square matrix."
    @assert length(ranks)==length(V) "reset! error: mismatched lengths of ranks ($(length(ranks))) and initial vectors ($(length(V)))."
    Q = zeros(eltype(gf), length(V), length(dimensions))
    iter = BandLanczosIterator(H, Block(V), length(dimensions)+length(V); keepvecs=false)
    fact = initialize(iter)
    dim = 0
    while dim <= length(dimensions)
        basis = fact.V
        for (i, b) in enumerate(basis)
            i += dim
            if i <= length(dimensions)
                for (j, v) in enumerate(V)
                    Q[j, i+dim] = dot(v, b)
                end
            end
        end
        dim = length(fact)
        dim<length(dimensions) && expand!(iter, fact)
    end
    E, U = eigen(Hermitian(rayleighquotient(fact)))
    gf.Q[ranks, dimensions] = Q*U
    for i = 1:length(dimensions)
        gf.E[dimensions[i]] = E[i]-E₀
    end
    return gf
end

"""
    GreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::Algorithm{<:ED}, sign::Bool=false;
        E₀::Union{Real, Nothing}=nothing, Ω::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, maxdim::Integer=200, kwargs...
    )
    GreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::ED, sign::Bool=false;
        E₀::Union{Real, Nothing}=nothing, Ω::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, maxdim::Integer=200, timer::TimerOutput=edtimer, kwargs...
    )

Construct a `GreenFunction`.
"""
@inline function GreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::Algorithm{<:ED}, sign::Bool=false;
        E₀::Union{Real, Nothing}=nothing, Ω::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, maxdim::Integer=200, kwargs...
    )
    return GreenFunction(operators, ed.frontend, sign; E₀, Ω, sector₀, maxdim, timer=ed.timer, kwargs...)
end
function GreenFunction(
    operators::AbstractVector{<:QuantumOperator}, ed::ED, sign::Bool=false;
    E₀::Union{Real, Nothing}=nothing, Ω::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, maxdim::Integer=200, timer::TimerOutput=edtimer, kwargs...
)
    if any(isnothing, (E₀, Ω, sector₀))
        eigensystem = eigen(ed; nev=1, timer=timer, kwargs...)
        E₀, Ω, sector₀ = only(eigensystem.values), only(eigensystem.vectors), only(eigensystem.sectors)
    end
    qn₀ = Abelian(sector₀)
    groups = Dict{typeof(sector₀), Vector{Int}}()
    for (i, operator) in enumerate(operators)
        sector = Sector(operator(qn₀), ed.system.hilbert; table=ed.matrixization.table)
        if haskey(groups, sector)
            push!(groups[sector], i)
        else
            groups[sector] = [i]
        end
    end
    total_dim = sum(sector->min(maxdim, dimension(sector)), keys(groups))
    result = GreenFunction(zeros(scalartype(Ω), length(operators), total_dim), zeros(real(scalartype(Ω)), total_dim), sign)
    offset = 0
    for (sector, ranks) in pairs(groups)
        local_dim = dimension(sector)
        if local_dim>0
            m = if (sector, sector) ∈ ed.matrixization.brakets
                value(only(matrix(ed, sector)))
            else
                EDMatrixization{scalartype(Ω)}(ed.matrixization.table, sector)(expand(ed.system))
            end
            V = [matrix(operators[index], (sector, sector₀), ed.matrixization.table)*Ω for index in ranks]
            reset!(result, m, V, E₀; ranks=ranks, dimensions=(offset+1):(offset+local_dim))
        end
    end
    return result
end

"""
    RetardedGreenFunction{T<:Number, V<:Real} <: AbstractGreenFunction{T}

Retarded Green's function obtained by Krylov space expansion.
"""
struct RetardedGreenFunction{T<:Number, V<:Real} <: AbstractGreenFunction{T}
    greater::GreenFunction{T, V}
    lesser::GreenFunction{T, V}
    sign::Bool
    function RetardedGreenFunction(greater::GreenFunction{T, V}, lesser::GreenFunction{T, V}, sign::Bool) where {T<:Number, V<:Real}
        @assert ~greater.sign "RetardedGreenFunction error: for greater Green's function, its `sign` must be `false`."
        @assert lesser.sign "RetardedGreenFunction error: for lesser Green's function, its `sign` must be `true`."
        @assert rank(greater)==rank(greater) "RetardedGreenFunction error: mismatched ranks of greater ($(rank(greater))) and lesser ($(rank(lesser))) Green's functions."
        new{T, V}(greater, lesser, sign)
    end
end

"""
    rank(gf::RetardedGreenFunction) -> Int

Get the rank of a `RetardedGreenFunction`.
"""
@inline rank(gf::RetardedGreenFunction) = rank(gf.greater)

"""
    (gf::RetardedGreenFunction)(dest::AbstractMatrix{<:Number}, ω::Number; sign::Bool=false) -> typeof(dest)

Get the values of a `RetardedGreenFunction` at `ω` and add the result to `dest`.
"""
function (gf::RetardedGreenFunction)(dest::AbstractMatrix{<:Number}, ω::Number; sign::Bool=false)
    gf.greater(dest, ω; sign=sign)
    gf.lesser(dest, ω; sign=xor(sign, gf.sign))
    return dest
end

"""
    RetardedGreenFunction(operators::AbstractVector{<:QuantumOperator}, ed::Algorithm{<:ED}, sign::Bool; maxdim::Integer=200, kwargs...)
    RetardedGreenFunction(operators::AbstractVector{<:QuantumOperator}, ed::ED, sign; maxdim::Integer=200, timer::TimerOutput=edtimer, kwargs...)

Construct a `RetardedGreenFunction`.
"""
@inline function RetardedGreenFunction(operators::AbstractVector{<:QuantumOperator}, ed::Algorithm{<:ED}, sign::Bool; maxdim::Integer=200, kwargs...)
    return RetardedGreenFunction(operators, ed.frontend, sign; maxdim=maxdim, timer=ed.timer, kwargs...)
end
function RetardedGreenFunction(operators::AbstractVector{<:QuantumOperator}, ed::ED, sign; maxdim::Integer=200, timer::TimerOutput=edtimer, kwargs...)
    eigensystem = eigen(ed; nev=1, timer=timer, kwargs...)
    E₀, Ω, sector₀ = only(eigensystem.values), only(eigensystem.vectors), only(eigensystem.sectors)
    greater = GreenFunction(operators, ed, false; E₀, Ω, sector₀, maxdim, timer)
    lesser = GreenFunction(map(adjoint, operators), ed, true; E₀, Ω, sector₀, maxdim, timer)
    return RetardedGreenFunction(greater, lesser, sign)
end
