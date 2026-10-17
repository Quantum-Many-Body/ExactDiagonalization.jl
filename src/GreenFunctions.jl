using Logging: AbstractLogger, Info, with_logger
import Logging: handle_message, min_enabled_level, shouldlog

"""
    RawStderrLogger <: AbstractLogger

A minimal logger that prints messages verbatim to `stderr` (no level tag, no metadata, no newline), used for the in-place (`\\r`-overwritten) progress display.
"""
struct RawStderrLogger <: AbstractLogger end
@inline min_enabled_level(::RawStderrLogger) = Info
@inline shouldlog(::RawStderrLogger, args...) = true
@inline function handle_message(::RawStderrLogger, level, msg, args...; kwargs...)
    print(stderr, msg)
    flush(stderr)
end

"""
    GreenFunctionKind{K}

The kind marker of a Green function construction, enabling dispatch on it: `GreenFunctionKind(:greater)`, `GreenFunctionKind(:lesser)` and `GreenFunctionKind(:retarded)` select the greater, lesser and retarded Green function, respectively.
"""
struct GreenFunctionKind{K} end
@inline function GreenFunctionKind(K::Symbol)
    @assert K∈(:greater, :lesser, :retarded) "GreenFunctionKind error: K must be `:greater`, `:lesser` or `:retarded`."
    return GreenFunctionKind{K}()
end
@inline Base.show(io::IO, ::GreenFunctionKind{K}) where {K} = print(io, K)
@inline Base.Symbol(::GreenFunctionKind{K}) where {K} = K

"""
    GreaterOrLesserKind

The union of `GreenFunctionKind(:greater)` and `GreenFunctionKind(:lesser)`: the legal `kind` constraint for the constructors of single-sided Green functions (see [`GreenFunctionKind`](@ref)).
"""
const GreaterOrLesserKind = Union{GreenFunctionKind{:greater}, GreenFunctionKind{:lesser}}

"""
    GreenFunctionMethod

Abstract type for the methods to compute Green's functions.
"""
abstract type GreenFunctionMethod end
@inline contenttoconfig(method::GreenFunctionMethod) = method

"""
    GreenFunction{T<:Number, V<:Real} <: Function

Green function expressed as a sum of simple poles,

    G[i, j](ω) = sum(L[i, k]*R[j, k]/(ω-E[k]) for k in eachindex(E)),

where `L` and `R` are the left and right amplitudes of each pole.

For Green's functions with a single initial state (e.g., the zero-temperature one), `R==conj(L)` and the pole amplitudes reduce to the single-amplitude-matrix form `L[i, k]*conj(L[j, k])`.
"""
struct GreenFunction{T<:Number, V<:Real} <: Function
    L::Matrix{T}
    R::Matrix{T}
    E::Vector{V}
    function GreenFunction(L::AbstractMatrix{<:Number}, R::AbstractMatrix{<:Number}, E::AbstractVector{<:Number})
        @assert size(L)==size(R) "GreenFunction error: mismatched L and R."
        @assert size(L, 2)==length(E) "GreenFunction error: mismatched L/R and E."
        new{promote_type(eltype(L), eltype(R)), eltype(E)}(L, R, E)
    end
end
@inline Base.:(==)(gf₁::GreenFunction, gf₂::GreenFunction) = ==(efficientoperations, gf₁, gf₂)
@inline Base.isequal(gf₁::GreenFunction, gf₂::GreenFunction) = isequal(efficientoperations, gf₁, gf₂)

"""
    eltype(gf::GreenFunction)
    eltype(::Type{<:GreenFunction{T}}) where {T<:Number}

Get the eltype of a `GreenFunction`.
"""
@inline Base.eltype(gf::GreenFunction) = eltype(typeof(gf))
@inline Base.eltype(::Type{<:GreenFunction{T}}) where {T<:Number} = T

"""
    size(gf::GreenFunction) -> NTuple{2, Int}

Get the size of a `GreenFunction`.
"""
@inline Base.size(gf::GreenFunction) = (rank(gf), rank(gf))

"""
    rank(gf::GreenFunction) -> Int

Get the rank of a `GreenFunction`.
"""
@inline rank(gf::GreenFunction) = size(gf.L, 1)

"""
    count(gf::GreenFunction) -> Int

Count the number of poles contributing to a `GreenFunction`.
"""
@inline Base.count(gf::GreenFunction) = size(gf.L, 2)

"""
    (gf::GreenFunction)(ω::Number) -> Matrix{promote_type(typeof(ω), eltype(gf))}

Get the values of a `GreenFunction` at `ω`.
"""
@inline (gf::GreenFunction)(ω::Number) = gf(zeros(promote_type(typeof(ω), eltype(gf)), size(gf)), ω)

"""
    (gf::GreenFunction)(dest::AbstractMatrix{<:Number}, ω::Number) -> typeof(dest)

Get the values of a `GreenFunction` at `ω` and add the result to `dest`.
"""
function (gf::GreenFunction)(dest::AbstractMatrix{<:Number}, ω::Number)
    for i = 1:rank(gf), j = 1:rank(gf)
        for k in 1:count(gf)
            dest[i, j] += gf.L[i, k] * gf.R[j, k] / (ω-gf.E[k])
        end
    end
    return dest
end

"""
    +(g₁::GreenFunction, g₂::GreenFunction) -> GreenFunction
    -(g₁::GreenFunction, g₂::GreenFunction) -> GreenFunction
    -(g::GreenFunction) -> GreenFunction

The algebra of Green functions: addition/subtraction is the concatenation of the pole sets (`E = [E₁; E₂]`, `L = [L₁ ±L₂]`, `R = [R₁ R₂]`), which is exact since `G(ω)` is linear in the poles. The ranks must match.
"""
function Base.:+(g₁::GreenFunction, g₂::GreenFunction)
    @assert rank(g₁)==rank(g₂) "+ error: mismatched ranks of the two Green's functions ($(rank(g₁)) vs $(rank(g₂)))."
    return GreenFunction([g₁.L g₂.L], [g₁.R g₂.R], [g₁.E; g₂.E])
end
@inline Base.:-(g::GreenFunction) = GreenFunction(-g.L, g.R, g.E)
function Base.:-(g₁::GreenFunction, g₂::GreenFunction)
    @assert rank(g₁)==rank(g₂) "- error: mismatched ranks of the two Green's functions ($(rank(g₁)) vs $(rank(g₂)))."
    L = [g₁.L g₂.L]
    L[:, count(g₁)+1:end] .*= -1
    return GreenFunction(L, [g₁.R g₂.R], [g₁.E; g₂.E])
end

"""
    GreenFunction(Q::AbstractMatrix{<:Number}, E::AbstractVector{<:Number})

Construct a `GreenFunction` from the single amplitude matrix `Q`, equivalent to `GreenFunction(Q, conj(Q), E)`.
"""
@inline GreenFunction(Q::AbstractMatrix{<:Number}, E::AbstractVector{<:Number}) = GreenFunction(Q, conj(Q), E)

"""
    GreenFunction(operators::AbstractVector{<:QuantumOperator}, ed::Algorithm{<:ED}, kind::GreenFunctionKind, method::GreenFunctionMethod=BandLanczosMethod(); kwargs...)

Construct a `GreenFunction` from a vector of quantum operators, an exact diagonalization algorithm, a kind marker ([`GreenFunctionKind`](@ref)), and a Green function method. The keyword arguments are passed to the underlying constructor.
"""
@inline function GreenFunction(operators::AbstractVector{<:QuantumOperator}, ed::Algorithm{<:ED}, kind::GreenFunctionKind, method::GreenFunctionMethod=BandLanczosMethod(); kwargs...)
    return GreenFunction(operators, ed.frontend, kind, method; timer=ed.timer, kwargs...)
end

## Zero-temperature Green functions
"""
    ExactMethod <: GreenFunctionMethod

Exact diagonalization method for zero-temperature GreenFunction.

Suitable for small Hilbert spaces and as the benchmark of [`BandLanczosMethod`](@ref).
"""
struct ExactMethod <: GreenFunctionMethod end
@inline Base.show(io::IO, ::ExactMethod) = print(io, "ExactMethod()")

"""
    BandLanczosMethod <: GreenFunctionMethod

Band Lanczos method for zero-temperature GreenFunction.
"""
struct BandLanczosMethod{OQ<:Orthogonalizer, OR<:Orthogonalizer} <: GreenFunctionMethod
    tol::Float64
    tolᵣ::Float64
    keepvecs::Bool
    maxdim::Int
    orthᵣ::OQ
    orthₒ::OR
end
@inline function Base.show(io::IO, method::BandLanczosMethod)
    print(io, "BandLanczosMethod(", method.tol, ", ", method.tolᵣ, ", ", method.keepvecs, ", ", method.maxdim, ", ", method.orthᵣ, ", ", method.orthₒ, ")")
end

"""
    BandLanczosMethod(
        orthᵣ::Orthogonalizer=ModifiedGramSchmidt2(), orthₒ::Orthogonalizer=ModifiedGramSchmidt();
        tol::Real=2e-8, tolᵣ::Real=1e-8, keepvecs::Bool=true, maxdim::Integer=200
    )

Construct a `BandLanczosMethod` for computing the zero-temperature Green function.

- `orthᵣ`: [`Orthogonalizer`](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.Orthogonalizer) used in `block_qr!` for rank determination of initial and residual blocks.
- `orthₒ`: [`Orthogonalizer`](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.Orthogonalizer) used in `block_reorthogonalize!` for reorthogonalization against the Krylov basis during the Lanczos recurrence.
- `tol`: Lanczos convergence tolerance — iteration stops when `normres < tol`.
- `tolᵣ`: QR rank determination tolerance in `block_qr!` — a vector with norm < `tolᵣ` is considered numerically zero.
"""
@inline function BandLanczosMethod(
    orthᵣ::Orthogonalizer=ModifiedGramSchmidt2(), orthₒ::Orthogonalizer=ModifiedGramSchmidt();
    tol::Real=2e-8, tolᵣ::Real=1e-8, keepvecs::Bool=true, maxdim::Integer=200
)
    return BandLanczosMethod(tol, tolᵣ, keepvecs, maxdim, orthᵣ, orthₒ)
end

### Zero-temperature greater/lesser Green function
"""
    GreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::GreaterOrLesserKind, method::Union{ExactMethod, BandLanczosMethod}=BandLanczosMethod();
        e₀::Union{Real, Nothing}=nothing, v₀::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
    )

Construct a zero-temperature `GreenFunction`, by full diagonalization of each target sector for `ExactMethod` and by the band Lanczos method for `BandLanczosMethod`.
"""
function GreenFunction(
    operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::GreaterOrLesserKind, method::Union{ExactMethod, BandLanczosMethod}=BandLanczosMethod();
    e₀::Union{Real, Nothing}=nothing, v₀::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
)
    @timeit timer string(kind) begin
        @info "Green Function ($(string(kind)))"
        if any(isnothing, (e₀, v₀, sector₀))
            eigensystem = eigen(ed; nev=1, timer=timer, kwargs...)
            @info "eigen complete"
            e₀, v₀, sector₀ = only(eigensystem.values), only(eigensystem.vectors), only(eigensystem.sectors)
        end
        T = promote_type(scalartype(operators), scalartype(ed))
        groups = sectorgroups(operators, ed, sector₀)
        blocks = Tuple{Matrix{T}, Matrix{T}, Vector{real(T)}, Vector{Int}}[]
        for (i, (sector, ranks)) in enumerate(pairs(groups))
            @info "($i/$(length(groups))) sector $(Abelian(sector))"
            if dimension(sector) > 0
                @timeit timer string(Abelian(sector)) begin
                    H = sparse(ed, sector; timer=timer)
                    @info "- matrix complete"
                    V = @timeit timer "initial states" [matrix(adjoint(operators)[index], (sector, sector₀), ed.matrixization.table, T)*v₀ for index in ranks]
                    @info "- initial states complete"
                    @timeit timer "poles" begin
                        Ẽ, S̃ = resolve(H, V, method)
                        Lₚ, Rₚ, Eₚ = poles(S̃, S̃, Ẽ, 1.0, e₀; kind=Symbol(kind))
                        push!(blocks, (Lₚ, Rₚ, Eₚ, ranks))
                    end
                    @info "- poles complete"
                end
            end
        end
        L, R, E = assemble(blocks, length(operators), T)
        return GreenFunction(L, R, E)
    end
end

"""
    sectorgroups(operators::AbstractVector{<:QuantumOperator}, ed::ED, sector₀::Sector) -> OrderedDict{typeof(sector₀), Vector{Int}}

Decompose the resolvent into independent sector blocks: `Aₐ†` maps a state in the ground-state sector `sector₀` with the quantum number `qn₀≣Abelian(sector₀)` into the target sector `Aₐ†(qn₀)`, and the Green function is block-diagonal in these target sectors. Return the operator ranks (indices into `operators`) grouped by their target sector, in order of first appearance.
"""
@inline function sectorgroups(operators::AbstractVector{<:QuantumOperator}, ed::ED, sector₀::Sector)
    groups = OrderedDict{typeof(sector₀), Vector{Int}}()
    qn₀ = Abelian(sector₀)
    for (i, operator) in enumerate(operators)
        sector = Sector(adjoint(operator)(qn₀), ed.system.hilbert; table=ed.matrixization.table)
        if haskey(groups, sector)
            push!(groups[sector], i)
        else
            groups[sector] = [i]
        end
    end
    return groups
end

"""
    resolve(H::AbstractMatrix{<:Number}, seeds::AbstractVector{<:AbstractVector{<:Number}}, method::ExactMethod) -> (E, S)

Exact variant of [`resolve`](@ref): the Ritz pairs are the full eigen pairs `{(E[k], ψₖ)}` of `H`, and the amplitudes are `S[a, k] = ⟨seedsₐ|ψₖ⟩`.
"""
function resolve(H::AbstractMatrix{<:Number}, V::AbstractVector{<:AbstractVector{<:Number}}, ::ExactMethod)
    E, U = eigen(Hermitian(collect(H)))
    return E, reduce(hcat, V)'*U
end

"""
    resolve(H::AbstractMatrix{<:Number}, seeds::AbstractVector{<:AbstractVector{<:Number}}, method::BandLanczosMethod) -> (Ẽ, S̃)

Resolve the band `seeds` against the Ritz pairs `{(Ẽ[k], ψ̃ₖ)}` of `H` obtained by the band Lanczos `method`, returning the Ritz energies `Ẽ` and the amplitudes `S̃[a, k] = ⟨seedsₐ|ψ̃ₖ⟩`.
"""
function resolve(H, V, method::BandLanczosMethod)
    maxdim = min(method.maxdim, size(H, 1))
    Q = zeros(promote_type(eltype(H), eltype(eltype(V))), length(V), maxdim)
    iter = BandLanczosIterator(H, Block(deepcopy(V)), maxdim+length(V), method.tol, method.tolᵣ, method.orthᵣ, method.orthₒ; keepvecs=method.keepvecs)
    fact = initialize(iter)
    offset = 0
    with_logger(RawStderrLogger()) do
        while true
            for (i, b) in enumerate(method.keepvecs ? fact.V[offset+1:end] : fact.V)
                i += offset
                if i <= maxdim
                    for (j, v) in enumerate(V)
                        Q[j, i] = dot(v, b)
                    end
                end
            end
            if length(fact) < maxdim && normres(fact) > iter.tol
                offset = length(fact)
                progress = offset / maxdim * 100
                @info "\r[ Info: - poles $(round(progress, digits=1))% ($offset/$maxdim)..."
                expand!(iter, fact)
            else
                offset>0 && @info "\n"
                break
            end
        end
    end
    ms = min(maxdim, length(fact))
    Ẽ, Ũ = eigen(Hermitian(rayleighquotient(fact)[1:ms, 1:ms]))
    return Ẽ, Q[:, 1:ms]*Ũ
end

"""
    poles(
        S̃::AbstractMatrix{<:Number}, D̃::AbstractMatrix{<:Number}, Ẽ::AbstractVector{<:Real}, weight::Number, ref::Real;
        kind::Symbol=:greater
    ) -> (L, R, E)

Assemble the poles of the resolvent for a **single** initial state with the given `weight` and reference energy `ref`, from the amplitudes `S̃[a, k] = ⟨seedₐ|ψ̃ₖ⟩` and `D̃[a, k] = ⟨probeₐ|ψ̃ₖ⟩` resolved against the (approximate) eigen pairs `{(Ẽ[k], ψ̃ₖ)}` of the target sector (see [`resolve`](@ref)). The pole at `k` has energy `±(Ẽ[k]-ref)` (`+` for `kind==:greater`, `-` otherwise) and left/right amplitudes

- `L[a, k] = weight * D̃[a, k]`, `R[a, k] = conj(S̃[a, k])` for `kind==:greater`;
- `L[a, k] = weight * conj(S̃[a, k])`, `R[a, k] = D̃[a, k]` for `kind==:lesser`.
"""
function poles(S̃::AbstractMatrix{<:Number}, D̃::AbstractMatrix{<:Number}, Ẽ::AbstractVector{<:Real}, weight::Number, ref::Real; kind::Symbol=:greater)
    @assert size(S̃)==size(D̃) && size(S̃, 2)==length(Ẽ) "poles error: mismatched amplitudes and energies."
    @assert kind∈(:greater, :lesser) "poles error: kind must be either `:greater` or `:lesser`."
    T = promote_type(eltype(S̃), eltype(D̃), typeof(weight))
    L = zeros(T, size(S̃)...)
    R = zeros(T, size(S̃)...)
    E = zeros(real(T), length(Ẽ))
    for k in axes(S̃, 2)
        E[k] = kind==:greater ? Ẽ[k]-ref : ref-Ẽ[k]
        for a in axes(S̃, 1)
            if kind == :greater
                L[a, k] = weight * D̃[a, k]
                R[a, k] = conj(S̃[a, k])
            else
                L[a, k] = weight * conj(S̃[a, k])
                R[a, k] = D̃[a, k]
            end
        end
    end
    return L, R, E
end

"""
    assemble(blocks, nops::Int, T) -> (L, R, E)

Assemble the pole blocks of the sector groups into the full pole arrays of a `GreenFunction`. Each block is a tuple `(Lₚ, Rₚ, Eₚ, ranks)` with the poles of one sector group, where `ranks` are the global operator indices of the group and `nops` is the total number of operators.
"""
@inline function assemble(blocks, nops::Int, T)
    total_npoles = sum(block->length(block[3]), blocks; init=0)
    L = zeros(T, nops, total_npoles)
    R = zeros(T, nops, total_npoles)
    E = zeros(real(T), total_npoles)
    offset = 0
    for (Lₚ, Rₚ, Eₚ, ranks) in blocks
        npoles = length(Eₚ)
        L[ranks, (offset+1):(offset+npoles)] = Lₚ
        R[ranks, (offset+1):(offset+npoles)] = Rₚ
        E[(offset+1):(offset+npoles)] = Eₚ
        offset += npoles
    end
    return L, R, E
end

### Zero-temperature Retarded Green function
"""
    GreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::GreenFunctionKind{:retarded}, method::Union{BandLanczosMethod, ExactMethod}=BandLanczosMethod();
        sign::Bool=false, e₀::Union{Real, Nothing}=nothing, v₀::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
    )

Construct the zero-temperature retarded Green function as the single merged pole sum of the greater and lesser parts (via the `+`/`-` algebra of `GreenFunction`), which are constructed by the zero-temperature `GreenFunction` constructor with the same `method` ([`BandLanczosMethod`](@ref) or [`ExactMethod`](@ref)) and share the same ground state.
"""
function GreenFunction(
    operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::GreenFunctionKind{:retarded}, method::Union{BandLanczosMethod, ExactMethod}=BandLanczosMethod();
    sign::Bool=false, e₀::Union{Real, Nothing}=nothing, v₀::Union{AbstractVector{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
)
    @timeit timer string(kind) begin
        @info "Retarded Green Function"
        if any(isnothing, (e₀, v₀, sector₀))
            eigensystem = eigen(ed; nev=1, timer=timer, kwargs...)
            @info "eigen complete"
            e₀, v₀, sector₀ = only(eigensystem.values), only(eigensystem.vectors), only(eigensystem.sectors)
        end
        greater = GreenFunction(operators, ed, GreenFunctionKind(:greater), method; e₀=e₀, v₀=v₀, sector₀=sector₀, timer=timer, kwargs...)
        lesser = GreenFunction(map(adjoint, operators), ed, GreenFunctionKind(:lesser), method; e₀=e₀, v₀=v₀, sector₀=sector₀, timer=timer, kwargs...)
        return sign ? greater - lesser : greater + lesser
    end
end

## Finite-temperature Green functions
"""
    ThermalExactMethod <: GreenFunctionMethod

Exact diagonalization method for the finite-temperature Green function. The trace is taken over the ground-state sector only, i.e. the canonical ensemble with all conserved quantum numbers fixed to the ground-state sector's values.

Suitable for small Hilbert spaces and as the benchmark of [`ThermalSeedBandLanczosMethod`](@ref).
"""
struct ThermalExactMethod <: GreenFunctionMethod
    β::Float64
    cutoff::Float64
    function ThermalExactMethod(β::Real, cutoff::Real)
        @assert β>0 "ThermalExactMethod error: inverse temperature β must be positive."
        @assert 0<=cutoff<=1 "ThermalExactMethod error: Boltzmann weight threshold cutoff must be in [0, 1]."
        return new(β, cutoff)
    end
end
@inline Base.show(io::IO, method::ThermalExactMethod) = print(io, "ThermalExactMethod(", method.β, ", ", method.cutoff, ")")

"""
    ThermalExactMethod(β::Real; cutoff::Real=0.0)

Construct a `ThermalExactMethod` for computing the finite-temperature Green function at inverse temperature `β` by exact diagonalization.

- `β`: the inverse temperature.
- `cutoff`: Boltzmann weight threshold (dimensionless). Only eigenstates with `exp(-β*(Eₙ-e₀))>=cutoff` are kept as initial states. The default `0.0` keeps all eigenstates (strictly exact).
"""
@inline ThermalExactMethod(β::Real; cutoff::Real=0.0) = ThermalExactMethod(β, cutoff)

"""
    ThermalSeedBandLanczosMethod{OQ<:Orthogonalizer, OR<:Orthogonalizer} <: GreenFunctionMethod

Thermally-weighted seed band Lanczos method for the finite-temperature Green function (the Kokalj variant of the low-temperature Lanczos method). The trace is taken over the ground-state sector only, i.e. the canonical ensemble with all conserved quantum numbers fixed to the ground-state sector's values.

For each of `nrandom` random vectors, a first Lanczos run prepares the approximate thermally-weighted pure state `|TPQ⟩=∑ₗexp(-βεₗ/2)⟨ψₗ|r⟩|ψₗ⟩` and a second band Lanczos run seeded by `A†|TPQ⟩` resolves the spectral function. Both runs keep the Krylov basis for full reorthogonalization.

The field `samples` is lazily initialized: empty at construction; overwritten by each `GreenFunction` construction with the Green functions of the individual random vectors (of the constructed kind, merged per sample for the `GreenFunctionKind(:retarded)` construction).
"""
struct ThermalSeedBandLanczosMethod{OQ<:Orthogonalizer, OR<:Orthogonalizer} <: GreenFunctionMethod
    β::Float64
    cutoff::Float64
    nrandom::Int
    maxdim₁::Int
    maxdim₂::Int
    seed::UInt
    tol::Float64
    tolᵣ::Float64
    orthᵣ::OQ
    orthₒ::OR
    samples::Vector{<:GreenFunction}
end
@inline function ThermalSeedBandLanczosMethod(β, cutoff, nrandom, maxdim₁, maxdim₂, seed, tol, tolᵣ, orthᵣ::OQ, orthₒ::OR) where {OQ<:Orthogonalizer, OR<:Orthogonalizer}
    return ThermalSeedBandLanczosMethod{OQ, OR}(β, cutoff, nrandom, maxdim₁, maxdim₂, seed, tol, tolᵣ, orthᵣ, orthₒ, GreenFunction[])
end
@inline function Base.show(io::IO, method::ThermalSeedBandLanczosMethod)
    print(io, "ThermalSeedBandLanczosMethod(", method.β, ", ", method.cutoff, ", ", method.nrandom, ", ", method.maxdim₁, ", ", method.maxdim₂, ", ", method.seed, ", ", method.tol, ", ", method.tolᵣ, ", ", method.orthᵣ, ", ", method.orthₒ, ")")
end
# For `ThermalSeedBandLanczosMethod`, the `samples` field is lazily filled at Green function construction, so it must not enter the fingerprint (a save would hash the filled one while a load would hash the empty one).
@inline contenttoconfig(method::ThermalSeedBandLanczosMethod) = (
    method.β, method.cutoff, method.nrandom, method.maxdim₁, method.maxdim₂, method.seed, method.tol, method.tolᵣ, method.orthᵣ, method.orthₒ
)

"""
    ThermalSeedBandLanczosMethod(
        orthᵣ::Orthogonalizer=ModifiedGramSchmidt2(), orthₒ::Orthogonalizer=ModifiedGramSchmidt();
        β::Real, cutoff::Real=1e-3, nrandom::Integer=10, maxdim₁::Integer=200, maxdim₂::Integer=200, seed::Integer=0, tol::Real=2e-8, tolᵣ::Real=1e-8
    )

Construct a `ThermalSeedBandLanczosMethod` for computing the finite-temperature Green function at inverse temperature `β`.

- `β`: the inverse temperature.
- `cutoff`: dimensionless weight threshold for the kept Lanczos eigenstates: only those with `exp(-β(εᵢ-e₀)/2)>=cutoff` are kept in the thermally-weighted seed, where `e₀` is the ground state energy. The discarded states bias the Green function by O(`cutoff`²) relatively.
- `nrandom`: number of random vectors.
- `maxdim₁`: number of Lanczos iterations of the first (state-preparing) run.
- `maxdim₂`: number of Lanczos iterations of the second (resolvent) run.
- `seed`: seed of the random number generator for reproducibility; `0` means not reseeding. For fully deterministic results, pass `e₀` and `sector₀` explicitly to the constructors, otherwise they are recomputed each call by a Lanczos eigen solver with a random initial vector, which may flip the signs of some LAPACK eigenvectors in the poles (the Green function values are invariant under these sign flips).
- `tol`: Lanczos convergence tolerance. Iteration stops when `normres < tol`.
- `tolᵣ`: QR rank determination tolerance in `block_qr!`. A vector with norm < `tolᵣ` is considered numerically zero.
- `orthᵣ`: [`Orthogonalizer`](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.Orthogonalizer) used in `block_qr!` for rank determination of initial and residual blocks.
- `orthₒ`: [`Orthogonalizer`](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.Orthogonalizer) used in `block_reorthogonalize!` for reorthogonalization against the Krylov basis during the Lanczos recurrence.
"""
@inline function ThermalSeedBandLanczosMethod(
    orthᵣ::Orthogonalizer=ModifiedGramSchmidt2(), orthₒ::Orthogonalizer=ModifiedGramSchmidt();
    β::Real, cutoff::Real=1e-3, nrandom::Integer=10, maxdim₁::Integer=200, maxdim₂::Integer=200, seed::Integer=0, tol::Real=2e-8, tolᵣ::Real=1e-8
)
    @assert β>0 "ThermalSeedBandLanczosMethod error: inverse temperature β must be positive."
    @assert 0<=cutoff<=1 "ThermalSeedBandLanczosMethod error: cutoff accuracy cutoff must be in [0, 1]."
    @assert nrandom>0 "ThermalSeedBandLanczosMethod error: nrandom must be positive."
    return ThermalSeedBandLanczosMethod(β, cutoff, nrandom, maxdim₁, maxdim₂, UInt64(seed), tol, tolᵣ, orthᵣ, orthₒ)
end

## Finite-temperature greater/lesser Green functions
"""
    GreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::GreaterOrLesserKind, method::ThermalExactMethod;
        E₀::Union{AbstractVector{<:Real}, Nothing}=nothing, U₀::Union{AbstractMatrix{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
    )

Construct the finite-temperature Green function at inverse temperature `method.β` by exact method. The trace is taken over the ground-state sector only, i.e. the canonical ensemble with all conserved quantum numbers fixed to the ground-state sector's values.

- `E₀` and `U₀`: the optional precomputed spectrum (all eigen values and eigen vectors) of the ground-state sector, which must be provided together; reused as provided (e.g. shared between the two kinds by the `GreenFunctionKind(:retarded)` constructor), and computed internally by dense diagonalization of `sector₀` when absent.
- `sector₀`: the ground-state sector; computed internally when absent. The Boltzmann weights are relative to its lowest eigen energy `E₀[1]`.
"""
function GreenFunction(
    operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::GreaterOrLesserKind, method::ThermalExactMethod;
    E₀::Union{AbstractVector{<:Real}, Nothing}=nothing, U₀::Union{AbstractMatrix{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
)
    @timeit timer string(kind) begin
        @info "Thermal Green Function ($(string(kind)), exact, β=$(method.β))"
        @assert isnothing(E₀)==isnothing(U₀) "GreenFunction error: E₀ and U₀ must be provided together."
        if isnothing(sector₀)
            eigensystem = eigen(ed; nev=1, timer=timer, kwargs...)
            @info "eigen complete"
            sector₀ = only(eigensystem.sectors)
        end
        if isnothing(E₀)
            E₀, U₀ = eigen(Hermitian(collect(sparse(ed, sector₀; timer=timer))))
            @info "- spectrum (sector₀) complete, dimension $(length(E₀))"
        end
        e₀ = E₀[1]
        w = @. exp(-method.β*(E₀-e₀))
        keep = findall(>=(method.cutoff), w)
        @assert length(keep)>0 "GreenFunction error: no eigenstate survives the Boltzmann weight threshold (cutoff=$(method.cutoff))."
        Z = sum(w)
        Eₖ, Uₖ, wₖ = E₀[keep], U₀[:, keep], w[keep]
        T = promote_type(scalartype(operators), scalartype(ed))
        ops′ = map(adjoint, operators)
        groups = sectorgroups(operators, ed, sector₀)
        blocks = Tuple{Matrix{T}, Matrix{T}, Vector{real(T)}, Vector{Int}}[]
        for (i, (sector, ranks)) in enumerate(pairs(groups))
            @info "($kind: $i/$(length(groups))) sector $(Abelian(sector))"
            dimension(sector) > 0 || continue
            @timeit timer string(Abelian(sector)) begin
                H = @timeit timer "matrix" sparse(ed, sector; timer=timer)
                @info "- matrix complete"
                eigensystem = @timeit timer "eigen" eigen(Hermitian(collect(H)))
                Xs = map(a->matrix(ops′[a], (sector, sector₀), ed.matrixization.table, T; cached=true), ranks)
                Lₚ, Rₚ, Eₚ = @timeit timer "poles" poles(Xs, Uₖ, eigensystem.vectors, eigensystem.values, wₖ./Z, Eₖ; kind=Symbol(kind))
                push!(blocks, (Lₚ, Rₚ, Eₚ, ranks))
            end
        end
        L, R, E = assemble(blocks, length(operators), T)
        return GreenFunction(L, R, E)
    end
end
"""
    GreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::GreaterOrLesserKind, method::ThermalSeedBandLanczosMethod;
        e₀::Union{Real, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
    )

Construct the finite-temperature Green function at inverse temperature `method.β` by the thermally-weighted seed band Lanczos method. The trace is taken over the ground-state sector only, i.e. the canonical ensemble with all conserved quantum numbers fixed to the ground-state sector's values.

The Green functions of the individual random vectors are stored in `method.samples` for statistical analysis (overwritten at each construction).

- `e₀`, `sector₀`: the ground-state energy and sector; recomputed each call when absent, by a Lanczos eigen solver with a random initial vector (see [`ThermalSeedBandLanczosMethod`](@ref) for the reproducibility implications).
"""
function GreenFunction(
    operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::GreaterOrLesserKind, method::ThermalSeedBandLanczosMethod;
    e₀::Union{Real, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
)
    @timeit timer string(kind) begin
        @info "Thermal Green Function ($(string(kind)), β=$(method.β), R=$(method.nrandom))"
        if any(isnothing, (e₀, sector₀))
            eigensystem = eigen(ed; nev=1, timer=timer, kwargs...)
            @info "eigen complete"
            e₀, sector₀ = only(eigensystem.values), only(eigensystem.sectors)
        end
        T = promote_type(scalartype(operators), scalartype(ed))
        H₀ = sparse(ed, sector₀; timer=timer)
        method.seed≠0 && seed!(method.seed)
        Z = zero(real(T))
        raws = GreenFunction{T, real(T)}[]
        empty!(method.samples)
        for r = 1:method.nrandom
            @info "random vector $r/$(method.nrandom)"
            sample = ThermalSample(H₀, method, e₀; timer=timer)
            Z += sample.z
            raw = GreenFunction(sample, operators, ed, kind, method; sector₀=sector₀, timer=timer)
            push!(raws, raw)
            push!(method.samples, GreenFunction(raw.L./sample.z, raw.R, raw.E))
        end
        return GreenFunction(mapreduce(r->r.L, hcat, raws)./Z, mapreduce(r->r.R, hcat, raws), mapreduce(r->r.E, vcat, raws))
    end
end

"""
    ThermalSample{T<:Number, V<:Real}

The per-sample data of the first Lanczos run (state preparation) of the thermally-weighted seed method (see [`ThermalSeedBandLanczosMethod`](@ref)) for a single random vector `|r⟩`, with `oᵢ = ⟨r|ψᵢ⟩` denoting the overlaps of the random vector with the Ritz states:

- `φ::Vector{T}`: the thermally-weighted state `|φ⟩ = ∑ᵢ exp(-βεᵢ/2)conj(oᵢ)|ψᵢ⟩` (the TPQ state);
- `energies::Vector{V}`: the Ritz energies `εᵢ` of the kept states (absolute energies);
- `states::Matrix{T}`: the kept Ritz states `|ψᵢ⟩`, stored as columns;
- `weights::Vector{T}`: the per-state weights `wᵢ = exp(-βεᵢ/2)oᵢ` (signed or complex, as they carry the overlaps `oᵢ`);
- `z::V`: the sample partition function `z = ∑ₗ |oₗ|²exp(-βεₗ)`.
"""
struct ThermalSample{T<:Number, V<:Real}
    φ::Vector{T}
    energies::Vector{V}
    states::Matrix{T}
    weights::Vector{T}
    z::V
    function ThermalSample(φ::AbstractVector{T}, energies::AbstractVector{V}, states::AbstractMatrix{T}, weights::AbstractVector{T}, z::V) where {T<:Number, V<:Real}
        @assert length(energies)==length(weights)==size(states, 2) && length(φ)==size(states, 1) "ThermalSample error: mismatched dimensions."
        return new{T, V}(φ, energies, states, weights, z)
    end
end

"""
    ThermalSample(H₀::AbstractMatrix{<:Number}, method::ThermalSeedBandLanczosMethod, e₀::Real; timer::TimerOutput=edtimer) -> ThermalSample

Construct a `ThermalSample` for a **single** random vector of the thermally-weighted seed method, by running the first Lanczos (state preparation) on the ground-state-sector Hamiltonian `H₀` with ground-state energy `e₀`. The random vector is drawn from the global RNG; call `seed!(method.seed)` before the sample loop for reproducibility.
"""
function ThermalSample(H₀::AbstractMatrix{<:Number}, method::ThermalSeedBandLanczosMethod, e₀::Real; timer::TimerOutput=edtimer)
    rv = randn(eltype(H₀), size(H₀, 1))
    normalize!(rv)
    iter = BandLanczosIterator(H₀, Block([rv]), method.maxdim₁+1, method.tol, method.tolᵣ, method.orthᵣ, method.orthₒ; keepvecs=true)
    fact = initialize(iter)
    ran = false
    @timeit timer "state preparation" with_logger(RawStderrLogger()) do
        while length(fact) < method.maxdim₁ && normres(fact) > iter.tol
            progress = length(fact) / method.maxdim₁ * 100
            @info "\r[ Info: - state preparation $(round(progress, digits=1))% ($(length(fact))/$(method.maxdim₁))..."
            ran = true
            expand!(iter, fact)
        end
    end
    ran && @info "\n"
    E, U = eigen(Hermitian(rayleighquotient(fact)))
    o = U[1, :]
    z = sum(abs2.(o) .* exp.(-method.β .* (E .- e₀)))
    keep = findall(i->exp(-method.β*(E[i]-e₀)/2)>=method.cutoff, eachindex(E))
    isempty(keep) && (keep = [argmin(E)])
    resmax = maximum(i->normres(fact)*abs(U[end, i]), keep; init=0.0)
    (E[1]-e₀>1e-8*max(1, abs(e₀)) || resmax>1e-6) && @warn "Possibly unconverged Lanczos states: E[1]-e₀=$(E[1]-e₀), maximum Ritz residual=$(resmax)."
    V = reduce(hcat, basis(fact))
    ψ = V * U[:, keep]
    w⁺ = @. exp(-method.β*(E[keep]-e₀)/2)
    φ = ψ * (w⁺ .* conj.(o[keep]))
    wᵢ = w⁺ .* o[keep]
    return ThermalSample(φ, E[keep], ψ, wᵢ, z)
end

"""
    poles(
        Xs::AbstractVector{<:AbstractMatrix}, states::AbstractMatrix{<:Number}, Ψ̃::AbstractMatrix{<:Number}, Ẽ::AbstractVector{<:Real}, weights::AbstractVector{<:Number}, refs::AbstractVector{<:Real};
        kind::Symbol=:greater, S̃::Union{Nothing, AbstractMatrix{<:Number}}=nothing
    ) -> (Lₚ, Rₚ, Eₚ)

Assemble the pole block of one target sector over **all** kept initial states, calling the single-state [`poles`](@ref) for each. `states` stores the kept initial states as columns, `Ψ̃` and `Ẽ` are the (approximate) eigen vectors and energies of the target sector, and `weights`/`refs` give the per-state weights and reference energies. The probe vectors are formed one state at a time, so that the full set of probes is never materialized. The seed amplitudes default to `S̃ = D̃` per state (seeds identical to probes, e.g. [`ThermalExactMethod`](@ref)); pass a fixed `S̃` (the amplitudes against the thermally-weighted seeds) for [`ThermalSeedBandLanczosMethod`](@ref).
"""
function poles(Xs::AbstractVector{<:AbstractMatrix}, states::AbstractMatrix{<:Number}, Ψ̃::AbstractMatrix{<:Number}, Ẽ::AbstractVector{<:Real}, weights::AbstractVector{<:Number}, refs::AbstractVector{<:Real}; kind::Symbol=:greater, S̃::Union{Nothing, AbstractMatrix{<:Number}}=nothing)
    T = promote_type(eltype(eltype(Xs)), eltype(states), eltype(Ψ̃), eltype(weights))
    isnothing(S̃) || (T = promote_type(T, eltype(S̃)))
    ns, ms = length(refs), length(Ẽ)
    Lₚ = zeros(T, length(Xs), ns*ms)
    Rₚ = zeros(T, length(Xs), ns*ms)
    Eₚ = zeros(real(T), ns*ms)
    for n = 1:ns
        D̃ₙ = reduce(hcat, [Xₐ*states[:, n] for Xₐ in Xs])'*Ψ̃
        Lₙ, Rₙ, Eₙ = poles(isnothing(S̃) ? D̃ₙ : S̃, D̃ₙ, Ẽ, weights[n], refs[n]; kind=kind)
        Lₚ[:, (n-1)*ms+1:n*ms] = Lₙ
        Rₚ[:, (n-1)*ms+1:n*ms] = Rₙ
        Eₚ[(n-1)*ms+1:n*ms] = Eₙ
    end
    return Lₚ, Rₚ, Eₚ
end

"""
    GreenFunction(sample::ThermalSample, operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::GreaterOrLesserKind, method::ThermalSeedBandLanczosMethod; sector₀::Sector, timer::TimerOutput=edtimer) -> GreenFunction

Construct the Green function for a **single** random vector (carried by `sample`) of the thermally-weighted seed method, in the mandatory ground-state sector `sector₀`. The operators are grouped by the sectors their adjoints map to from `sector₀` (see [`sectorgroups`](@ref)), and the Hamiltonian and operator matrices of each sector are constructed on the fly inside the loop (cheaply, via the disk caches). The constructed `GreenFunction` is **unnormalized** and serves as a data container: divide its `L` by `sample.z` for the Green function of the individual sample, or accumulate the poles over the samples and divide by the total partition function for the combined Green function.
"""
function GreenFunction(sample::ThermalSample, operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::GreaterOrLesserKind, method::ThermalSeedBandLanczosMethod; sector₀::Sector, timer::TimerOutput=edtimer)
    T = promote_type(scalartype(operators), scalartype(ed))
    ops′ = map(adjoint, operators)
    groups = sectorgroups(operators, ed, sector₀)
    blocks = Tuple{Matrix{T}, Matrix{T}, Vector{real(T)}, Vector{Int}}[]
    for (sector, ranks) in pairs(groups)
        dimension(sector) > 0 || continue
        @timeit timer string(Abelian(sector)) begin
            H = @timeit timer "matrix" sparse(ed, sector; timer=timer)
            Xs = @timeit timer "operator matrices" map(a->matrix(ops′[a], (sector, sector₀), ed.matrixization.table, T; cached=true), ranks)
            @timeit timer "poles" begin
                W = [Xₐ*sample.φ for Xₐ in Xs]
                Ẽ, Ψ̃ = resolve(H, W, method)
                S̃ = reduce(hcat, W)'*Ψ̃
                L, R, E = poles(Xs, sample.states, Ψ̃, Ẽ, sample.weights, sample.energies; kind=Symbol(kind), S̃=S̃)
                push!(blocks, (L, R, E, ranks))
            end
        end
    end
    return GreenFunction(assemble(blocks, length(operators), T)...)
end

"""
    resolve(H::AbstractMatrix{<:Number}, seeds::AbstractVector{<:AbstractVector{<:Number}}, method::ThermalSeedBandLanczosMethod) -> (Ẽ, Ψ̃)

Run the second band Lanczos (resolvent resolution) of the thermally-weighted seed method, seeded by `seeds` (the operators acting on the thermally-weighted state), returning the Ritz values `Ẽ` and the Ritz vectors `Ψ̃` of `H` as columns of a dense matrix. The amplitudes against the individual initial states are **not** resolved here; they are computed afterwards from `Ψ̃`, one state at a time, so that the full set of probe vectors is never materialized. The Krylov basis is always kept, as required by the deferred resolution.
"""
function resolve(H::AbstractMatrix{<:Number}, seeds::AbstractVector{<:AbstractVector{<:Number}}, method::ThermalSeedBandLanczosMethod)
    maxdim = min(method.maxdim₂, size(H, 1))
    iter = BandLanczosIterator(H, Block(deepcopy(seeds)), maxdim+length(seeds), method.tol, method.tolᵣ, method.orthᵣ, method.orthₒ; keepvecs=true)
    fact = initialize(iter)
    with_logger(RawStderrLogger()) do
        while length(fact) < maxdim && normres(fact) > iter.tol
            offset = length(fact)
            progress = offset / maxdim * 100
            @info "\r[ Info: - poles $(round(progress, digits=1))% ($offset/$maxdim)..."
            expand!(iter, fact)
        end
    end
    ms = min(maxdim, length(fact))
    B = reduce(hcat, basis(fact)[1:ms])
    Ẽ, Ũ = eigen(Hermitian(rayleighquotient(fact)[1:ms, 1:ms]))
    return Ẽ, B*Ũ
end

### Finite-temperature Retarded Green functions
"""
    GreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::GreenFunctionKind{:retarded}, method::ThermalExactMethod;
        sign::Bool=false, E₀::Union{AbstractVector{<:Real}, Nothing}=nothing, U₀::Union{AbstractMatrix{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
    )

Construct the finite-temperature retarded Green function at inverse temperature `method.β` by the exact method, as the single merged pole sum of the greater and lesser parts (via the `+`/`-` algebra of `GreenFunction`), which share the same ground-state sector spectrum: computed internally, or reused from `E₀` and `U₀` provided together.
"""
function GreenFunction(
    operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::GreenFunctionKind{:retarded}, method::ThermalExactMethod;
    sign::Bool=false, E₀::Union{AbstractVector{<:Real}, Nothing}=nothing, U₀::Union{AbstractMatrix{<:Number}, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
)
    @timeit timer string(kind) begin
        @info "Retarded Green Function (exact, β=$(method.β))"
        @assert isnothing(E₀)==isnothing(U₀) "GreenFunction error: E₀ and U₀ must be provided together."
        if isnothing(sector₀)
            eigensystem = eigen(ed; nev=1, timer=timer, kwargs...)
            @info "eigen complete"
            sector₀ = only(eigensystem.sectors)
        end
        if isnothing(E₀)
            E₀, U₀ = eigen(Hermitian(collect(sparse(ed, sector₀; timer=timer))))
            @info "- spectrum (sector₀) complete, dimension $(length(E₀))"
        end
        greater = GreenFunction(operators, ed, GreenFunctionKind(:greater), method; E₀=E₀, U₀=U₀, sector₀=sector₀, timer=timer, kwargs...)
        lesser = GreenFunction(map(adjoint, operators), ed, GreenFunctionKind(:lesser), method; E₀=E₀, U₀=U₀, sector₀=sector₀, timer=timer, kwargs...)
        return sign ? greater - lesser : greater + lesser
    end
end

"""
    GreenFunction(
        operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::GreenFunctionKind{:retarded}, method::ThermalSeedBandLanczosMethod;
        sign::Bool=false, e₀::Union{Real, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
    )

Construct the finite-temperature retarded Green function at inverse temperature `method.β` by the thermally-weighted seed band Lanczos method, as the single merged pole sum of the greater and lesser parts (via the `+`/`-` algebra of `GreenFunction`), which share the same per-sample data of the first Lanczos runs: each random vector is used for both parts (exactly paired), and the merged per-sample Green functions are stored in `method.samples`.
"""
function GreenFunction(
    operators::AbstractVector{<:QuantumOperator}, ed::ED, kind::GreenFunctionKind{:retarded}, method::ThermalSeedBandLanczosMethod;
    sign::Bool=false, e₀::Union{Real, Nothing}=nothing, sector₀::Union{Sector, Nothing}=nothing, timer::TimerOutput=edtimer, kwargs...
)
    @timeit timer string(kind) begin
        @info "Retarded Green Function (β=$(method.β), R=$(method.nrandom))"
        if any(isnothing, (e₀, sector₀))
            eigensystem = eigen(ed; nev=1, timer=timer, kwargs...)
            @info "eigen complete"
            e₀, sector₀ = only(eigensystem.values), only(eigensystem.sectors)
        end
        T = promote_type(scalartype(operators), scalartype(ed))
        H₀ = sparse(ed, sector₀; timer=timer)
        method.seed≠0 && seed!(method.seed)
        Z = zero(real(T))
        gs = GreenFunction{T, real(T)}[]
        ls = GreenFunction{T, real(T)}[]
        empty!(method.samples)
        for r = 1:method.nrandom
            @info "random vector $r/$(method.nrandom)"
            sample = ThermalSample(H₀, method, e₀; timer=timer)
            Z += sample.z
            g = GreenFunction(sample, operators, ed, GreenFunctionKind(:greater), method; sector₀=sector₀, timer=timer)
            l = GreenFunction(sample, map(adjoint, operators), ed, GreenFunctionKind(:lesser), method; sector₀=sector₀, timer=timer)
            push!(gs, g)
            push!(ls, l)
            sg = GreenFunction(g.L./sample.z, g.R, g.E)
            sl = GreenFunction(l.L./sample.z, l.R, l.E)
            push!(method.samples, sign ? sg - sl : sg + sl)
        end
        greater = GreenFunction(mapreduce(x->x.L, hcat, gs)./Z, mapreduce(x->x.R, hcat, gs), mapreduce(x->x.E, vcat, gs))
        lesser = GreenFunction(mapreduce(x->x.L, hcat, ls)./Z, mapreduce(x->x.R, hcat, ls), mapreduce(x->x.E, vcat, ls))
        return sign ? greater - lesser : greater + lesser
    end
end
