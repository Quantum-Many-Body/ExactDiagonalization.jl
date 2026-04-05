using Logging: AbstractLogger, Info, with_logger
import Logging: handle_message, min_enabled_level, shouldlog

"""
    GreenFunctionMethod

Abstract type for methods used to reset GreenFunction.
"""
abstract type GreenFunctionMethod end

"""
    BandLanczosMethod <: GreenFunctionMethod

Band Lanczos method for GreenFunction.
"""
struct BandLanczosMethod <: GreenFunctionMethod
    tol::Float64
    keepvecs::Bool
    maxdim::Int
end
@inline BandLanczosMethod(; tol::Real=1e-10, keepvecs::Bool=false, maxdim::Integer=200) = BandLanczosMethod(tol, keepvecs, maxdim)

"""
    ExactMethod <: GreenFunctionMethod

Exact diagonalization method for GreenFunction.
"""
struct ExactMethod <: GreenFunctionMethod end

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

Get the dimension of the Krylov space expanded to obtain a `GreenFunction`.
"""
@inline dimension(gf::GreenFunction) = size(gf.Q, 2)

"""
    (gf::GreenFunction)(dest::AbstractMatrix{<:Number}, ω::Number; sign::Bool=false) -> typeof(dest)

Get the values of a `GreenFunction` at `ω` and add the result to `dest` (when `sign` is `false`) or subtract the result from `dest` (when `sign` is `true`).
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
    set!(
        gf::GreenFunction, H::AbstractMatrix{<:Number}, V::AbstractVector{<:AbstractVector{<:Number}}, e₀::Real, method::GreenFunctionMethod=BandLanczosMethod();
        kind::Symbol=:greater, ranks::AbstractVector{<:Integer}=1:rank(gf), dimensions::AbstractVector{<:Integer}=1:dimension(gf)
    ) -> typeof(gf)

Reset (a block) of `GreenFunction`.

Here, `method` can be either an instance of [`BandLanczosMethod`](@ref) or [`ExactMethod`](@ref).
"""
@inline function set!(
    gf::GreenFunction, H::AbstractMatrix{<:Number}, V::AbstractVector{<:AbstractVector{<:Number}}, e₀::Real, method::GreenFunctionMethod=BandLanczosMethod();
    kind::Symbol=:greater, ranks::AbstractVector{<:Integer}=1:rank(gf), dimensions::AbstractVector{<:Integer}=1:dimension(gf)
)
    @assert allequal(size(H)) "set! error: input Hamiltonian ($(join(size(H), "×"))) is not a square matrix."
    @assert length(ranks)==length(V) "set! error: mismatched lengths of ranks ($(length(ranks))) and initial vectors ($(length(V)))."
    @assert kind∈(:greater, :lesser) "set! error: kind must be either `:greater` or `lesser`."
    fill!(view(gf.Q, ranks, dimensions), 0)
    fill!(view(gf.E, dimensions), 0)
    Q, E, U, dimensions = qeu(H, V, dimensions, method)
    if kind == :greater
        gf.Q[ranks, dimensions] = Q * U
        broadcast!(-, view(gf.E, dimensions), E, e₀)
    else
        gf.Q[ranks, dimensions] = conj!(Q*U)
        broadcast!(-, view(gf.E, dimensions), e₀, E)
    end
    return gf
end
@inline function qeu(H, V, dimensions, method::BandLanczosMethod)
    Q = zeros(ComplexF64, length(V), length(dimensions))
    iter = BandLanczosIterator(H, Block(deepcopy(V)), length(dimensions)+length(V), method.tol; keepvecs=method.keepvecs)
    fact = initialize(iter)
    offset = 0
    total_dim = length(dimensions)
    with_logger(RawStderrLogger()) do
        while true
            for (i, b) in enumerate(method.keepvecs ? fact.V[offset+1:end] : fact.V)
                i += offset
                if i <= length(dimensions)
                    for (j, v) in enumerate(V)
                        Q[j, i] = dot(v, b)
                    end
                end
            end
            if length(fact)<length(dimensions) && normres(fact)>iter.tol
                offset = length(fact)
                progress = offset / total_dim * 100
                @info "\r[ Info: - set! $(round(progress, digits=1))% ($offset/$total_dim)..."
                expand!(iter, fact)
            else
                offset>0 && @info "\n"
                break
            end
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
    return Q, E, U, dimensions
end
@inline function qeu(H, V, dimensions, ::ExactMethod)
    E, U = eigen(Hermitian(collect(H)))
    Q = zeros(ComplexF64, length(V), length(dimensions))
    for (i, v) in enumerate(V)
        Q[i, :] = v
        conj!(view(Q, i, :))
    end
    return Q, E, U, dimensions
end
struct RawStderrLogger <: AbstractLogger end
@inline min_enabled_level(::RawStderrLogger) = Info
@inline shouldlog(::RawStderrLogger, args...) = true
@inline function handle_message(::RawStderrLogger, level, msg, args...; kwargs...)
    print(stderr, msg)
    flush(stderr)
end

"""
    GreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::Algorithm{<:ED}, method::GreenFunctionMethod=BandLanczosMethod();
        kind::Symbol=:greater, e₀::Union{Real, Nothing}=nothing, v₀::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, kwargs...
    )
    GreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::ED, method::GreenFunctionMethod=BandLanczosMethod();
        kind::Symbol=:greater, e₀::Union{Real, Nothing}=nothing, v₀::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
    )

Construct a `GreenFunction`.
"""
@inline function GreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::Algorithm{<:ED}, method::GreenFunctionMethod=BandLanczosMethod();
        kind::Symbol=:greater, e₀::Union{Real, Nothing}=nothing, v₀::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, kwargs...
    )
    return GreenFunction(operators, ed.frontend, method; kind, e₀, v₀, sector₀, timer=ed.timer, kwargs...)
end
function GreenFunction(
    operators::AbstractVector{<:QuantumOperator}, ed::ED, method::GreenFunctionMethod=BandLanczosMethod();
    kind::Symbol=:greater, e₀::Union{Real, Nothing}=nothing, v₀::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
)
    @timeit timer string(kind) begin
        @info "Green Function ($(string(kind)))"
        if any(isnothing, (e₀, v₀, sector₀))
            eigensystem = eigen(ed; nev=1, timer=timer, kwargs...)
            @info "eigen complete"
            e₀, v₀, sector₀ = only(eigensystem.values), only(eigensystem.vectors), only(eigensystem.sectors)
        end
        maxdim = method isa BandLanczosMethod ? method.maxdim : typemax(Int)
        qn₀ = Abelian(sector₀)
        groups = OrderedDict{typeof(sector₀), Vector{Int}}()
        for (i, operator) in enumerate(operators)
            sector = Sector(adjoint(operator)(qn₀), ed.system.hilbert; table=ed.matrixization.table)
            if haskey(groups, sector)
                push!(groups[sector], i)
            else
                groups[sector] = [i]
            end
        end
        total_dim = sum(sector->min(maxdim, dimension(sector)), keys(groups))::Int
        result = GreenFunction(zeros(scalartype(ed), length(operators), total_dim), zeros(real(scalartype(ed)), total_dim))
        offset = 0
        for (i, (sector, ranks)) in enumerate(pairs(groups))
            local_dim = min(maxdim, dimension(sector))
            @info "($i/$(length(groups))) sector $(Abelian(sector))"
            if local_dim > 0
                @timeit timer string(Abelian(sector)) begin
                    m = if (sector, sector) ∈ ed.matrixization.brakets
                        matrix(ed, sector; timer)
                    else
                        @timeit timer "matrix" EDMatrixization{scalartype(ed)}(ed.matrixization.table, sector)(expand(ed.system))
                    end
                    @info "- matrix complete"
                    T = promote_type(scalartype(operators), scalartype(ed))
                    V = @timeit timer "initial states" [matrix(adjoint(operators)[index], (sector, sector₀), ed.matrixization.table, T)*v₀ for index in ranks]
                    @info "- initial states complete"
                    @timeit timer "set!" set!(result, value(only(m)), V, e₀, method; kind=kind, ranks=ranks, dimensions=(offset+1):(offset+local_dim))
                    @info "- set! complete"
                    offset += local_dim
                end
            end
        end
        return result
    end
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
        @assert rank(greater)==rank(lesser) "RetardedGreenFunction error: mismatched ranks of greater ($(rank(greater))) and lesser ($(rank(lesser))) Green's functions."
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
    RetardedGreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::Algorithm{<:ED}, method::GreenFunctionMethod=BandLanczosMethod();
        sign::Bool=false, e₀::Union{Real, Nothing}=nothing, v₀::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, kwargs..
    )
    RetardedGreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::ED, method::GreenFunctionMethod=BandLanczosMethod();
        sign::Bool=false, e₀::Union{Real, Nothing}=nothing, v₀::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
    )

Construct a `RetardedGreenFunction`.
"""
@inline function RetardedGreenFunction(
    operators::AbstractVector{<:QuantumOperator}, ed::Algorithm{<:ED}, method::GreenFunctionMethod=BandLanczosMethod();
    sign::Bool=false, e₀::Union{Real, Nothing}=nothing, v₀::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, kwargs...
)
    return RetardedGreenFunction(operators, ed.frontend, method; sign=sign, e₀=e₀, v₀=v₀, sector₀=sector₀, timer=ed.timer, kwargs...)
end
function RetardedGreenFunction(
    operators::AbstractVector{<:QuantumOperator}, ed::ED, method::GreenFunctionMethod=BandLanczosMethod();
    sign::Bool=false, e₀::Union{Real, Nothing}=nothing, v₀::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
)
    @timeit timer "RetardedGreenFunction" begin
        @info "Retarded Green Function"
        if any(isnothing, (e₀, v₀, sector₀))
            eigensystem = eigen(ed; nev=1, timer=timer, kwargs...)
            @info "eigen complete"
            e₀, v₀, sector₀ = only(eigensystem.values), only(eigensystem.vectors), only(eigensystem.sectors)
        end
        greater = GreenFunction(operators, ed, method; kind=:greater, e₀=e₀, v₀=v₀, sector₀=sector₀, timer=timer)
        lesser = GreenFunction(map(adjoint, operators), ed, method; kind=:lesser, e₀=e₀, v₀=v₀, sector₀=sector₀, timer=timer)
        return RetardedGreenFunction(greater, lesser, sign)
    end
end
