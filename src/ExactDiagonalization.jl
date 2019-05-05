module ExactDiagonalization

using Reexport: @reexport

include("FED.jl")

@reexport using .FED

end # module
