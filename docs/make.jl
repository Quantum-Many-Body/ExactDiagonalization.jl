push!(LOAD_PATH,"../src/")

using Documenter
using DocumenterTools
using ExactDiagonalization

makedocs(
    format=     Documenter.HTML(
                    prettyurls= !("local" in ARGS),
                    canonical=  "https://quantum-many-body.github.io/ExactDiagonalization.jl/latest/",
                    assets=     ["assets/favicon.ico"],
                    analytics=  "UA-89508993-1",
                    ),
    clean=      false,
    sitename=   "ExactDiagonalization.jl",
    pages=      [
                "Home"      =>  "index.md",
                "Manul"     =>  [
                    "man/FED.md",
                    ]
                ]
)

deploydocs(
    repo=       "github.com/Quantum-Many-Body/ExactDiagonalization.jl.git",
    target=     "build",
    deps=       nothing,
    make=       nothing,
)
