using Documenter
using ExactDiagonalization

DocMeta.setdocmeta!(ExactDiagonalization, :DocTestSetup, :(using ExactDiagonalization); recursive=true)

makedocs(;
    modules=[ExactDiagonalization],
    authors="waltergu <waltergu1989@gmail.com> and contributors",
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
            "examples/HubbardModel.md",
        ],
        "Manual" => [
            "man/EDCore.md",
            "man/CanonicalFockSystems.md",
            "man/CanonicalSpinSystems.md",
            "man/GreenFunctions.md",
        ]
    ]
)

deploydocs(;
    repo="github.com/Quantum-Many-Body/ExactDiagonalization.jl.git"
)
