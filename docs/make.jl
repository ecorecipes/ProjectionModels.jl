using Documenter, StructuredPopulationCore

makedocs(;
    modules = [StructuredPopulationCore],
    warnonly = true,
    authors = "Simon Frost",
    sitename = "StructuredPopulationCore.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ecorecipes.github.io/StructuredPopulationCore.jl",
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => [
            "Types" => "api/types.md",
            "State Domains & Blocks" => "api/state_blocks.md",
            "Analysis" => "api/analysis.md",
            "Eigenanalysis" => "api/eigenanalysis.md",
            "Matrix Properties" => "api/properties.md",
            "Time-Lag Support" => "api/time_lag.md",
            "Utilities" => "api/utilities.md",
        ],
    ],
)

deploydocs(;
    repo = "github.com/ecorecipes/StructuredPopulationCore.jl.git",
)
