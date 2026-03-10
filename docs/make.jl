using Documenter, ProjectionModels

makedocs(;
    modules = [ProjectionModels],
    warnonly = true,
    authors = "Simon Frost",
    sitename = "ProjectionModels.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ecorecipes.github.io/ProjectionModels.jl",
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => [
            "Types" => "api/types.md",
            "Analysis" => "api/analysis.md",
            "Eigenanalysis" => "api/eigenanalysis.md",
            "Matrix Properties" => "api/properties.md",
            "Time-Lag Support" => "api/time_lag.md",
            "Utilities" => "api/utilities.md",
        ],
    ],
)

deploydocs(;
    repo = "github.com/ecorecipes/ProjectionModels.jl.git",
)
