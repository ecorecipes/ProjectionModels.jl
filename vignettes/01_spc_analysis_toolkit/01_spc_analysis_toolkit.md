# SPC: The Shared Analysis Layer


## Introduction

`StructuredPopulationCore.jl` (SPC) provides the analytical backbone
shared by every package in the population-dynamics ecosystem. Its
functions accept:

- **Raw matrices** (`AbstractMatrix`)
- **`MatrixProjectionModel`** objects (from `MatrixProjectionModels.jl`)
- **`IPMSolution`** objects (from `IntegralProjectionModels.jl`)

This means the same code — `lambda`, `stable_distribution`,
`sensitivity`, `elasticity` — works regardless of whether the underlying
model was specified as a matrix projection model, an integral projection
model, or something else entirely.

This vignette demonstrates this **dispatch uniformity** by analysing the
same biological system through three different representations.

``` julia
using StructuredPopulationCore
using MatrixProjectionModels
using IntegralProjectionModels
using LinearAlgebra

const SPC = StructuredPopulationCore
const MPM = MatrixProjectionModels
const IPM = IntegralProjectionModels
```

    IntegralProjectionModels

## A simple perennial plant

We define vital rates for a size-structured perennial herb:

``` julia
# Vital-rate parameters
β0_s, β1_s = -2.0, 0.6     # survival logistic
β0_g, β1_g = 0.4, 0.9      # growth regression
σ_g = 0.4                   # growth SD
repr_int = -3.0             # flowering logistic intercept
repr_slope = 1.2            # flowering logistic slope
seed_int = 0.0              # log seed count intercept
seed_slope = 0.4            # log seed count slope
μ_r, σ_r = 1.0, 0.3        # recruit size distribution
L, U_bound = 0.0, 5.0      # domain bounds
```

    (0.0, 5.0)

## Three representations of one model

### 1. IPM solution object

``` julia
domain = ContinuousDomain(L, U_bound, 80)
surv = IPM.LinearSurvival(β0_s, β1_s)
grow = IPM.NormalGrowth(β0_g, β1_g, σ_g)
fec  = IPM.LogisticFecundityRate(repr_int, repr_slope, seed_int, seed_slope, μ_r, σ_r)
K    = IPM.PKernel(surv, grow, domain; eviction = IPM.TruncatedDistributions) +
       IPM.FKernel(fec, domain)
n0   = IPM.normal_population(domain, 2.0, 0.6)
prob = IPM.IPMProblem(SPC.SimpleIPM(), SPC.DensityIndependent(),
                      SPC.Deterministic(), K, domain, n0, (0, 1))

ipm_sol = IPM.solve(prob, IPM.EigenAnalysis())
println("IPM solution type: ", typeof(ipm_sol))
```

    IPM solution type: IPMSolution{Vector{Int64}, Vector{Vector{Float64}}, Matrix{Float64}, @NamedTuple{lambda::Float64, stable_dist::Vector{Float64}, repro_value::Vector{Float64}}}

### 2. Matrix Projection Model (from the same kernel)

``` julia
A_matrix = ipm_sol.kernel_matrices   # 80×80 discretised kernel
mpm = MPM.MatrixProjectionModel(A_matrix)
println("MPM type: ", typeof(mpm))
```

    MPM type: MatrixProjectionModel{Float64, Matrix{Float64}}

### 3. Raw matrix (plain Julia array)

``` julia
raw = Matrix(mpm)   # plain Matrix{Float64}
println("Raw matrix type: ", typeof(raw))
```

    Raw matrix type: Matrix{Float64}

## SPC analysis: one API, three inputs

### Dominant eigenvalue (λ)

``` julia
λ_ipm = SPC.lambda(ipm_sol)
λ_mpm = SPC.lambda(mpm)
λ_raw = SPC.lambda(raw)

println("λ (IPM solution): ", round(λ_ipm; digits = 10))
println("λ (MPM object):   ", round(λ_mpm; digits = 10))
println("λ (raw matrix):   ", round(λ_raw; digits = 10))
println()
println("All agree: ", λ_ipm ≈ λ_mpm ≈ λ_raw)
```

    λ (IPM solution): 0.6683660542
    λ (MPM object):   0.6683660542
    λ (raw matrix):   0.6683660542

    All agree: true

### Stable distribution (right eigenvector $\mathbf{w}$)

``` julia
w_ipm = SPC.stable_distribution(ipm_sol)
w_mpm = SPC.stable_distribution(mpm)
w_raw = SPC.stable_distribution(raw)

println("max |w_ipm − w_mpm| = ", maximum(abs.(w_ipm .- w_mpm)))
println("max |w_mpm − w_raw| = ", maximum(abs.(w_mpm .- w_raw)))
println("∑w = ", round(sum(w_raw); digits = 6), " (normalised to 1)")
```

    max |w_ipm − w_mpm| = 0.0
    max |w_mpm − w_raw| = 0.0
    ∑w = 1.0 (normalised to 1)

### Reproductive value (left eigenvector $\mathbf{v}$)

``` julia
v_ipm = SPC.reproductive_value(ipm_sol)
v_mpm = SPC.reproductive_value(mpm)
v_raw = SPC.reproductive_value(raw)

println("max |v_ipm − v_mpm| = ", maximum(abs.(v_ipm .- v_mpm)))
println("max |v_mpm − v_raw| = ", maximum(abs.(v_mpm .- v_raw)))
```

    max |v_ipm − v_mpm| = 0.0
    max |v_mpm − v_raw| = 0.0

### Sensitivity and elasticity

``` julia
sens_ipm = SPC.sensitivity(ipm_sol)
sens_mpm = SPC.sensitivity(mpm)
sens_raw = SPC.sensitivity(raw)

elas_ipm = SPC.elasticity(ipm_sol)
elas_mpm = SPC.elasticity(mpm)
elas_raw = SPC.elasticity(raw)

println("Sensitivity agreement:")
println("  max |IPM − MPM| = ", maximum(abs.(sens_ipm .- sens_mpm)))
println("  max |MPM − raw| = ", maximum(abs.(sens_mpm .- sens_raw)))
println()
println("Elasticity agreement:")
println("  max |IPM − MPM| = ", maximum(abs.(elas_ipm .- elas_mpm)))
println("  max |MPM − raw| = ", maximum(abs.(elas_mpm .- elas_raw)))
println("  ∑ elasticity    = ", round(sum(elas_raw); digits = 6), " (should be 1)")
```

    Sensitivity agreement:
      max |IPM − MPM| = 0.0
      max |MPM − raw| = 0.0

    Elasticity agreement:
      max |IPM − MPM| = 0.0
      max |MPM − raw| = 0.0
      ∑ elasticity    = 1.0 (should be 1)

### Damping ratio

``` julia
ρ_mpm = SPC.damping_ratio(mpm)
ρ_raw = SPC.damping_ratio(raw)

println("Damping ratio:")
println("  MPM: ", round(ρ_mpm; digits = 6))
println("  Raw: ", round(ρ_raw; digits = 6))
println("  (Higher = faster convergence to stable distribution)")
```

    Damping ratio:
      MPM: 1.266421
      Raw: 1.266421
      (Higher = faster convergence to stable distribution)

### Structural properties

``` julia
println("Matrix structural properties:")
println("  Irreducible: ", SPC.is_irreducible(raw))
println("  Primitive:   ", SPC.is_primitive(raw))
println("  Ergodic:     ", SPC.is_ergodic(raw))
```

    Matrix structural properties:
      Irreducible: true
      Primitive:   false
      Ergodic:     false

## What SPC provides vs. what’s package-specific

| Function | SPC | MPM | IPM | Input types |
|----|----|----|----|----|
| `lambda` | ✓ | ✓ (delegates) | ✓ (delegates) | Matrix, MPM, IPMSolution |
| `stable_distribution` | ✓ | ✓ | ✓ | Matrix, MPM, IPMSolution |
| `reproductive_value` | ✓ | ✓ | ✓ | Matrix, MPM, IPMSolution |
| `sensitivity` | ✓ | ✓ | ✓ | Matrix, MPM, IPMSolution |
| `elasticity` | ✓ | ✓ | ✓ | Matrix, MPM, IPMSolution |
| `damping_ratio` | ✓ | ✓ | — | Matrix, MPM |
| `is_irreducible` | ✓ | — | — | Matrix |
| `is_primitive` | ✓ | — | — | Matrix |
| `net_repro_rate` | — | ✓ | — | matU, matR |
| `gen_time` | — | ✓ | — | matU, matR |

**Key design principle:** SPC is the shared analytical core. MPM and IPM
re-export its generic functions and extend them for their specific
types. Functions that require matrix decomposition ($R_0$, generation
time) live in MPM because they need the $\mathbf{U}$ / $\mathbf{F}$ /
$\mathbf{C}$ split.

## Summary

`StructuredPopulationCore.jl` ensures that **population analysts write
analysis code once** and apply it to any representation:

``` julia
# This works for ALL three:
λ = SPC.lambda(model_or_solution)
w = SPC.stable_distribution(model_or_solution)
S = SPC.sensitivity(model_or_solution)
E = SPC.elasticity(model_or_solution)
```

The underlying model can be a 2×2 Leslie matrix, an 80×80 discretised
IPM kernel, or the output of an IPM solve — the analysis layer doesn’t
care.

## References

- Caswell, H. (2001). *Matrix Population Models*. Sinauer.
- Ellner, S. P., Childs, D. Z., & Rees, M. (2016). *Data-driven
  Modelling of Structured Populations*. Springer.
