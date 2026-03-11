module ExactDiagonalizationPlotsExt

using RecipesBase: @recipe
using ExactDiagonalization: SpinCoherentStateProjection
using QuantumLattices: Assignment, str

"""
    @recipe plot(projection::Assignment{<:SpinCoherentStateProjection})

Define the recipe for the visualization of a spin coherent state projection assignment.
"""
@recipe function plot(projection::Assignment{<:SpinCoherentStateProjection})
    title --> str(projection)
    titlefontsize --> 10
    seriestype := :heatmap
    xlabel --> "θ/π"
    ylabel --> "φ/π"
    projection.data.polars/pi, projection.data.azimuths/pi, projection.data.values
end

end # module
