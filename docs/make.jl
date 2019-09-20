using Documenter, XFields

makedocs(;
    modules=[XFields],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/EthanAnderes/XFields.jl/blob/{commit}{path}#L{line}",
    sitename="XFields.jl",
    authors="Ethan Anderes",
    assets=String[],
)

deploydocs(;
    repo="github.com/EthanAnderes/XFields.jl",
)
