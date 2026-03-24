module ExactDiagonalizationMakieExt

import Makie
using QuantumLattices: Assignment, str
using ExactDiagonalization: SpinCoherentStateProjection

function Makie.plot!(ax::Makie.AbstractAxis, projection::Assignment{<:SpinCoherentStateProjection}; kwargs...)
    ax.title = str(projection)
    ax.titlesize = 16
    ax.xlabel = "θ/π"
    ax.ylabel = "φ/π"
    data = projection.data.values
    x = projection.data.polars ./ pi
    y = projection.data.azimuths ./ pi
    Δx, Δy = length(x) > 1 ? (x[2]-x[1], y[2]-y[1]) : (one(eltype(x)), one(eltype(y)))
    Makie.xlims!(ax, get(kwargs, :xlims, (x[1]-Δx, x[end]+Δx))...)
    Makie.ylims!(ax, get(kwargs, :ylims, (y[1]-Δy, y[end]+Δy))...)
    _set_axis_properties!(ax, kwargs)
    return Makie.heatmap!(
        ax, x, y, transpose(data);
        colorrange=get(kwargs, :clims, extrema(data)),
        _filter_kwargs(kwargs, Makie.Heatmap)...
    )
end
@inline _filter_kwargs(kwargs, plot_type::Type) = NamedTuple(k => v for (k, v) in kwargs if k in Makie.attribute_names(plot_type))
@inline _set_axis_properties!(ax, kwargs) = for (k, v) in pairs(kwargs) hasproperty(ax, k) && setproperty!(ax, k, v) end

end # module
