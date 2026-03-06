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
    (gf::AbstractGreenFunction)(ω::Number) -> Matrix{promote_type(typeof(ω), eltype(gf))}

Get the values of an `AbstractGreenFunction` at `ω`.
"""
@inline (gf::AbstractGreenFunction)(ω::Number) = gf(zeros(promote_type(typeof(ω), eltype(gf)), size(gf)), ω)

"""
    GreenFunction{T<:Number, V<:Real} <: AbstractGreenFunction{T}

Green function obtained by Krylov space expansion.
"""
struct GreenFunction{T<:Number, V<:Real} <: AbstractGreenFunction{T}
    Q::Matrix{T}
    E::Vector{V}
    function GreenFunction(Q::AbstractMatrix{<:Number}, E::AbstractVector{<:Number})
        @assert size(Q, 2)==length(E) "GreenFunction error: mismatched Q and E."
        new{eltype(Q), eltype(E)}(Q, E)
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

Get the values of a `GreenFunction` at `ω` and add the result to `dest` (when `sign` is `false`) or substrate the result from `dest` (when `sign` is `true`).
"""
function (gf::GreenFunction)(dest::AbstractMatrix{<:Number}, ω::Number; sign::Bool=false)
    factor = sign ? -1 : 1
    for i = 1:rank(gf), j = 1:rank(gf)
        for k in 1:dimension(gf)
            coeff = factor / (ω-gf.E[k])
            dest[i, j] += coeff * gf.Q[i, k] * conj(gf.Q[j, k])
        end
    end
    return dest
end

"""
    reset!(
        gf::GreenFunction, H::AbstractMatrix{<:Number}, V::AbstractVector{<:AbstractVector{<:Number}}, E₀::Real, kind::Symbol;
        ranks::AbstractVector{<:Integer}=1:rank(gf), dimensions::AbstractVector{<:Integer}=1:dimension(gf), tol::Real=1e-12
    ) -> typeof(gf)

Reset (a block) of `GreenFunction`.
"""
function reset!(
    gf::GreenFunction, H::AbstractMatrix{<:Number}, V::AbstractVector{<:AbstractVector{<:Number}}, E₀::Real, kind::Symbol;
    ranks::AbstractVector{<:Integer}=1:rank(gf), dimensions::AbstractVector{<:Integer}=1:dimension(gf), tol::Real=1e-12
)
    @assert allequal(size(H)) "reset! error: input Hamiltonian ($(join(size(H), "×"))) is not a square matrix."
    @assert length(ranks)==length(V) "reset! error: mismatched lengths of ranks ($(length(ranks))) and initial vectors ($(length(V)))."
    @assert kind∈(:greater, :lesser) "reset! error: kind must be either `:greater` or `lesser`."
    Q = zeros(eltype(gf), length(V), length(dimensions))
    iter = BandLanczosIterator(H, Block(deepcopy(V)), length(dimensions)+length(V), tol; keepvecs=false)
    fact = initialize(iter)
    offset = 0
    while true
        basis = fact.V
        for (i, b) in enumerate(basis)
            i += offset
            if i <= length(dimensions)
                for (j, v) in enumerate(V)
                    Q[j, i] = dot(v, b)
                end
            end
        end
        if length(fact)<length(dimensions) && normres(fact)>iter.tol
            offset = length(fact)
            expand!(iter, fact)
        else
            break
        end
    end
    M = rayleighquotient(fact)
    if length(fact)<length(dimensions)
        Q = Q[:, 1:length(fact)]
        dimensions = dimensions[1:length(fact)]
    elseif length(fact)>length(dimensions)
        M = M[1:length(dimensions), 1:length(dimensions)]
    end
    E, U = eigen(Hermitian(M))
    if kind == :greater
        gf.Q[ranks, dimensions] = Q * U
        broadcast!(-, view(gf.E, dimensions), E, E₀)
    else
        gf.Q[ranks, dimensions] = conj!(Q*U)
        broadcast!(-, view(gf.E, dimensions), E₀, E)
    end
    return gf
end

"""
    GreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::Algorithm{<:ED}, kind::Symbol=:greater;
        E₀::Union{Real, Nothing}=nothing, Ω::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, maxdim::Integer=200, tol::Real=1e-12, kwargs...
    )
    GreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::Symbol=:greater;
        E₀::Union{Real, Nothing}=nothing, Ω::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, maxdim::Integer=200, tol::Real=1e-12, timer::TimerOutput=edtimer, kwargs...
    )

Construct a `GreenFunction`.
"""
@inline function GreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::Algorithm{<:ED}, kind::Symbol=:greater;
        E₀::Union{Real, Nothing}=nothing, Ω::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, maxdim::Integer=200, tol::Real=1e-12, kwargs...
    )
    return GreenFunction(operators, ed.frontend, kind; E₀, Ω, sector₀, maxdim, tol, timer=ed.timer, kwargs...)
end
function GreenFunction(
    operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::Symbol=:greater;
    E₀::Union{Real, Nothing}=nothing, Ω::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, maxdim::Integer=200, tol::Real=1e-12, timer::TimerOutput=edtimer, kwargs...
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
    result = GreenFunction(zeros(scalartype(Ω), length(operators), total_dim), zeros(real(scalartype(Ω)), total_dim))
    offset = 0
    for (sector, ranks) in pairs(groups)
        local_dim = min(maxdim, dimension(sector))
        if local_dim>0
            m = if (sector, sector) ∈ ed.matrixization.brakets
                matrix(ed, sector)
            else
                EDMatrixization{scalartype(Ω)}(ed.matrixization.table, sector)(expand(ed.system))
            end
            V = [matrix(operators[index], (sector, sector₀), ed.matrixization.table)*Ω for index in ranks]
            reset!(result, value(only(m)), V, E₀, kind; ranks=ranks, dimensions=(offset+1):(offset+local_dim), tol=tol)
            offset += local_dim
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
    (gf::RetardedGreenFunction)(dest::AbstractMatrix{<:Number}, ω::Number) -> typeof(dest)

Get the values of a `RetardedGreenFunction` at `ω` and add the result to `dest`.
"""
function (gf::RetardedGreenFunction)(dest::AbstractMatrix{<:Number}, ω::Number)
    gf.greater(dest, ω)
    gf.lesser(dest, ω; sign=gf.sign)
    return dest
end

"""
    RetardedGreenFunction(operators::AbstractVector{<:QuantumOperator}, ed::Algorithm{<:ED}, sign::Bool=false; maxdim::Integer=200, tol::Real=1e-12, kwargs...)
    RetardedGreenFunction(operators::AbstractVector{<:QuantumOperator}, ed::ED, sign::Bool=false; maxdim::Integer=200, tol::Real=1e-12, timer::TimerOutput=edtimer, kwargs...)

Construct a `RetardedGreenFunction`.
"""
@inline function RetardedGreenFunction(operators::AbstractVector{<:QuantumOperator}, ed::Algorithm{<:ED}, sign::Bool=false; maxdim::Integer=200, tol::Real=1e-12, kwargs...)
    return RetardedGreenFunction(operators, ed.frontend, sign; maxdim=maxdim, tol=tol, timer=ed.timer, kwargs...)
end
function RetardedGreenFunction(operators::AbstractVector{<:QuantumOperator}, ed::ED, sign::Bool=false; maxdim::Integer=200, tol::Real=1e-12, timer::TimerOutput=edtimer, kwargs...)
    eigensystem = eigen(ed; nev=1, timer=timer, kwargs...)
    E₀, Ω, sector₀ = only(eigensystem.values), only(eigensystem.vectors), only(eigensystem.sectors)
    greater = GreenFunction(operators, ed, :greater; E₀, Ω, sector₀, maxdim, tol, timer)
    lesser = GreenFunction(map(adjoint, operators), ed, :lesser; E₀, Ω, sector₀, maxdim, tol, timer)
    return RetardedGreenFunction(greater, lesser, sign)
end
