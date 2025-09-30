"""
    const basicoptions = (
        nev = "number of eigenvalues to be computed",
        which = "type of eigenvalues to be computed",
        tol = "tolerance of the computation",
        maxiter = "maximum iteration of the computation",
        v₀ = "initial state",
        krylovdim = "maximum dimension of the Krylov subspace that will be constructed",
        verbosity = "verbosity level"
    )

Basic options of actions for exact diagonalization method.
"""
const basicoptions = (
    nev = "number of eigenvalues to be computed",
    which = "type of eigenvalues to be computed",
    tol = "tolerance of the computation",
    maxiter = "maximum iteration of the computation",
    v₀ = "initial state",
    krylovdim = "maximum dimension of the Krylov subspace that will be constructed",
    verbosity = "verbosity level",
    release = "release or not the cache for the construction of the sparse matrix representation of the Hamiltonian"
)

"""
    EDEigen{S<:Tuple{Vararg{Union{Abelian, Sector}}}} <: Action

Eigen system by exact diagonalization method.
"""
struct EDEigen{S<:Tuple{Vararg{Union{Abelian, Sector}}}} <: Action
    sectors::S
end
@inline options(::Type{<:Assignment{<:EDEigen}}) = basicoptions

"""
    EDEigen(sectors::Union{Abelian, Sector}...)
    EDEigen(sectors:::Tuple{Vararg{Union{Abelian, Sector}}})

Construct an `EDEigen`.
"""
@inline EDEigen(sectors::Union{Abelian, Sector}...) = EDEigen(sectors)
@inline run!(ed::Algorithm{<:ED}, edeigen::Assignment{<:EDEigen}; options...) = eigen(ed, edeigen.action.sectors...; options...)

"""
    GroundStateExpectation{D<:Number, O<:Array{<:Union{Operator, Operators}}} <: Action

Ground state expectation of operators.
"""
struct GroundStateExpectation{D<:Number, O<:Array{<:Union{Operator, Operators}}} <: Action
    operators::O
    GroundStateExpectation{D}(operators::Array{<:Union{Operator, Operators}}) where {D<:Number} = new{D, typeof(operators)}(operators)
end
@inline options(::Type{<:Assignment{<:GroundStateExpectation}}) = basicoptions
@inline GroundStateExpectation(operators::Array{<:Union{Operator, Operators}}) = GroundStateExpectation{Float64}(operators)

"""
    GroundStateExpectationData{A<:Array{<:Number}} <: Data

Data of ground state expectation of operators, including:

`values::A`: values of the ground state expectation.
"""
struct GroundStateExpectationData{A<:Array{<:Number}} <: Data
    values::A
end
function run!(ed::Algorithm{<:ED}, expectation::Assignment{<:GroundStateExpectation{D}}; options...) where {D<:Number}
    eigensystem = only(expectation.dependencies)
    @assert isa(eigensystem, Assignment{<:EDEigen}) "run! error: wrong dependencies."
    table = ed.frontend.matrixization.table
    Ω = only(eigensystem.data.vectors)
    sector = only(eigensystem.data.sectors)
    result = zeros(D, size(expectation.action.operators))
    for (i, operator) in enumerate(expectation.action.operators)
        result[i] = dot(Ω, matrix(operator, (sector, sector), table), Ω)
    end
    return GroundStateExpectationData(result)
end

"""
    StaticTwoPointCorrelator{O<:Union{Operator, Operators}, R<:ReciprocalSpace} <: Action

Static two-point correlation function.
"""
struct StaticTwoPointCorrelator{O<:Union{Operator, Operators}, R<:ReciprocalSpace} <: Action
    operators::Matrix{O}
    reciprocalspace::R
end
@inline options(::Type{<:Assignment{<:StaticTwoPointCorrelator}}) = basicoptions

"""
    StaticTwoPointCorrelatorData{R<:ReciprocalSpace, V<:Array{Float64}} <: Data

Data of static two-point correlation function, including:

1) `reciprocalspace::R`: reciprocal space to compute the static two-point correlation function.
2) `values::V`: values of the static two-point correlation function.
"""
struct StaticTwoPointCorrelatorData{R<:ReciprocalSpace, V<:Array{Float64}} <: Data
    reciprocalspace::R
    values::V
end
function run!(ed::Algorithm{<:ED}, correlator::Assignment{<:StaticTwoPointCorrelator}; options...)
    eigensystem = only(correlator.dependencies)
    @assert isa(eigensystem, Assignment{<:EDEigen}) "run! error: wrong dependencies."
    lattice = ed.frontend.lattice
    operators = correlator.action.operators
    @assert size(operators)==(length(lattice), length(lattice)) "run! error: the size ($(join(size(operators), "x"))) of the operators doest not match the length ($(length(lattice))) of the lattice."
    table = ed.frontend.matrixization.table
    Ω = only(eigensystem.data.vectors)
    sector = only(eigensystem.data.sectors)
    len = length(correlator.action.reciprocalspace)
    result = initialization(correlator.action.reciprocalspace)
    for index in CartesianIndices(operators)
        factor = dot(Ω, matrix(operators[index], (sector, sector), table), Ω)
        r = lattice[index[2]] - lattice[index[1]]
        for (k, momentum) in enumerate(correlator.action.reciprocalspace)
            # Note we always use e^ikr to perform the Fourier transformation, which may cause problems if the static correlator is complex.
            # However, we only consider real static correlators for now.
            phase = exp(1im*dot(momentum, r))
            result[k] += real(factor*phase)/len
        end
    end
    return StaticTwoPointCorrelatorData(correlator.action.reciprocalspace, result)
end
@inline initialization(reciprocalspace::Union{BrillouinZone, ReciprocalZone}) = zeros(Float64, map(length, reverse(shape(reciprocalspace)))...)
@inline initialization(reciprocalspace::ReciprocalSpace) = zeros(Float64, length(reciprocalspace), 1)

"""
    SpinCoherentState <: CompositeDict{Int, Tuple{Float64, Float64}}

Spin coherent state on a block of lattice sites.

The structure of the spin coherent state is specified by a `Dict{Int, Tuple{Float64, Float64}}`, which contains the site-(θ, ϕ) pairs with site being the site index in a lattice and (θ, ϕ) denoting the polar and azimuth angles in radians of the classical magnetic moment on this site.
"""
struct SpinCoherentState <: CompositeDict{Int, Tuple{Float64, Float64}}
    structure::Dict{Int, Tuple{Float64, Float64}}
end
@inline getcontent(state::SpinCoherentState, ::Val{:contents}) = state.structure

"""
    SpinCoherentState(structure::AbstractDict{Int, <:AbstractVector{<:Number}})
    SpinCoherentState(structure::AbstractDict{Int, <:NTuple{2, Number}}; unit::Symbol=:radian)

Construct a spin coherent state on a block of lattice sites.
"""
function SpinCoherentState(structure::AbstractDict{Int, <:AbstractVector{<:Number}})
    new = Dict{Int, NTuple{2, Float64}}()
    for (site, direction) in structure
        new[site] = (polar(direction), azimuth(direction))
    end
    return SpinCoherentState(new)
end
function SpinCoherentState(structure::AbstractDict{Int, <:NTuple{2, Number}}; unit::Symbol=:degree)
    @assert unit∈(:degree, :radian) "SpinCoherentState error: unit must be either `:degree` or `:radian`."
    new = Dict{Int, NTuple{2, Float64}}()
    for (site, (θ, ϕ)) in structure
        if unit==:degree
            θ = deg2rad(θ)
            ϕ = deg2rad(ϕ)
        end
        new[site] = (θ, ϕ)
    end
    return SpinCoherentState(new)
end

"""
    (state::SpinCoherentState)(bases::AbelianBases, table::AbstractDict, dtype=ComplexF64) -> Vector{dtype}

Get the vector representation of a spin coherent state with the given Abelian bases and table.
"""
function (state::SpinCoherentState)(bases::AbelianBases, table::AbstractDict, dtype=ComplexF64)
    angles = permute!(collect(values(state)), sortperm(collect(keys(state)), by=site->table[𝕊(site, :α)]))
    vs = [zeros(dtype, dimension(graded)) for graded in bases.locals]
    for ((θ, ϕ), v) in zip(angles, vs)
        S = (length(v)-1)//2
        v[1] = one(dtype)
        v[:] = exp(-1im*ϕ*matrix(𝕊{S}('z'))) * exp(-1im*θ*matrix(𝕊{S}('y'))) * v
    end
    intermediate = eltype(vs)[]
    for positions in bases.partition
        for (i, position) in enumerate(positions)
            if i==1
                push!(intermediate, vs[position])
            else
                intermediate[end] = kron(intermediate[end], vs[position])
            end
        end
    end
    result = intermediate[1]
    for i = 2:length(intermediate)
        result = kron(result, intermediate[i])
    end
    isa(Abelian(bases), ℤ₁) || (result = result[range(bases)])
    return result
end

"""
    SpinCoherentStateProjection <: Action

Projection of states obtained by exact diagonalization method onto spin coherent states.
"""
struct SpinCoherentStateProjection <: Action
    configuration::SpinCoherentState
    polars::Vector{Float64}
    azimuths::Vector{Float64}
end
@inline options(::Type{<:Assignment{<:SpinCoherentStateProjection}}) = basicoptions

"""
    SpinCoherentStateProjection(configuration::SpinCoherentState, polars::AbstractVector{<:Real}, azimuths::AbstractVector{<:Real})
    SpinCoherentStateProjection(configuration::SpinCoherentState, np::Integer, na::Integer)

Construct a `SpinCoherentStateProjection`.
"""
@inline function SpinCoherentStateProjection(configuration::SpinCoherentState, np::Integer, na::Integer)
    return SpinCoherentStateProjection(configuration, range(0, pi, np), range(0, 2pi, na))
end

"""
    SpinCoherentStateProjectionData <: Data

Data of spin coherent state projection, including:

1) `polars::Vector{Float64}`: global polar angles of the spin coherent states.
2) `azimuths::Vector{Float64}`: global azimuth angles of the spin coherent states.
3) `values::Matrix{Float64}`: projection of the state obtained by exact diagonalization method onto the spin coherent states.
"""
struct SpinCoherentStateProjectionData <: Data
    polars::Vector{Float64}
    azimuths::Vector{Float64}
    values::Matrix{Float64}
end
function run!(ed::Algorithm{<:ED}, projection::Assignment{<:SpinCoherentStateProjection}; options...)
    @assert length(ed.frontend.lattice)==length(projection.action.configuration) "run! error: mismatched lattice and magnetic moment configuration."
    eigensystem = only(projection.dependencies)
    @assert isa(eigensystem, Assignment{<:EDEigen}) "run! error: wrong dependencies."
    table = ed.frontend.matrixization.table
    Ω = only(eigensystem.data.vectors)
    sector = only(eigensystem.data.sectors)
    Ψ = empty(projection.action.configuration)
    result = zeros(Float64, length(projection.action.azimuths), length(projection.action.polars))
    for (i, θ) in enumerate(projection.action.polars)
        for (j, ϕ) in enumerate(projection.action.azimuths)
            for (site, (θ₀, ϕ₀)) in projection.action.configuration
                Ψ[site] = (θ+θ₀, ϕ+ϕ₀)
            end
            result[j, i] = abs2(dot(Ψ(sector, table), Ω))
        end
    end
    return SpinCoherentStateProjectionData(projection.action.polars, projection.action.azimuths, result)
end
@recipe function plot(projection::Assignment{<:SpinCoherentStateProjection})
    title --> str(projection)
    titlefontsize --> 10
    seriestype := :heatmap
    xlabel --> "θ/π"
    ylabel --> "φ/π"
    projection.data.polars/pi, projection.data.azimuths/pi, projection.data.values
end
