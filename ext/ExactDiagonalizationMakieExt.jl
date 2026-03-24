module ExactDiagonalizationMakieExt

import Makie
using QuantumLattices: Assignment, str
using ExactDiagonalization: SpinCoherentStateProjection

function Makie.plot!(ax::Makie.AbstractAxis, projection::Assignment{<:SpinCoherentStateProjection}; kwargs...)
    ax.title = get(kwargs, :title, str(projection))
    ax.titlesize = get(kwargs, :titlesize, 16)
    ax.xlabel = get(kwargs, :xlabel, "θ/π")
    ax.ylabel = get(kwargs, :ylabel, "φ/π")
    polars = projection.data.polars
    azimuths = projection.data.azimuths
    data = projection.data.values
    x = polars ./ pi
    y = azimuths ./ pi
    Δx, Δy = length(x) > 1 ? (x[2] - x[1], y[2] - y[1]) : (one(eltype(x)), one(eltype(y)))
    Makie.xlims!(ax, get(kwargs, :xlims, (x[1] - Δx, x[end] + Δx))...)
    Makie.ylims!(ax, get(kwargs, :ylims, (y[1] - Δy, y[end] + Δy))...)
    clims = get(kwargs, :clims, extrema(data))
    return Makie.heatmap!(ax, x, y, transpose(data); colorrange=clims)
end

end # module
