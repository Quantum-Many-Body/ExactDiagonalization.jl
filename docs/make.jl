using Documenter
using ExactDiagonalization

DocMeta.setdocmeta!(ExactDiagonalization, :DocTestSetup, :(using ExactDiagonalization); recursive=true)

makedocs(;
    modules=[ExactDiagonalization],
    authors="waltergu <waltergu1989@gmail.com> and contributors",
    repo="https://github.com/Quantum-Many-Body/ExactDiagonalization.jl/blob/{commit}{path}#{line}",
    sitename="ExactDiagonalization.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Quantum-Many-Body.github.io/ExactDiagonalization.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "examples/Introduction.md",
            ]
        ]
)

deploydocs(;
    repo="github.com/Quantum-Many-Body/ExactDiagonalization.jl.git"
)
