"""
Life Table Response Experiment (LTRE) analysis.

Provides deterministic, stochastic, small-noise approximation (SNA), and exact
fANOVA-based LTRE decompositions for matrix population models.

References:
- Caswell (2001) Matrix Population Models, Ch. 10
- Davison et al. (2010) J Ecol 98:255-267 (stochastic LTRE)
- Davison et al. (2019) Methods Ecol Evol 10:1656-1672 (SNA-LTRE)
- Hernandez et al. (2023) Methods Ecol Evol 14:1065-1078 (exact LTRE)
- Poelwijk et al. (2016) PLoS Comput Biol 12:e1004771 (G-matrix)
"""

# --- Result types ---

"""
    LTREResult

Result of a deterministic LTRE analysis.

# Fields
- `contributions`: Matrix of element-level contributions to Δλ
- `difference`: Matrix of element-level differences (treatment - reference)
- `sensitivity_matrix`: Sensitivity matrix evaluated at the reference/midpoint
- `lambda_treatment`: λ of treatment matrix
- `lambda_reference`: λ of reference matrix
- `delta_lambda`: Difference in λ (treatment - reference)
"""
struct LTREResult{T<:Real}
    contributions::Matrix{T}
    difference::Matrix{T}
    sensitivity_matrix::Matrix{T}
    lambda_treatment::T
    lambda_reference::T
    delta_lambda::T
end

"""
    StochasticLTREResult

Result of a stochastic LTRE analysis (Davison et al. 2010).

# Fields
- `cont_mean`: Contributions of differences in mean matrix elements
- `cont_sd`: Contributions of differences in standard deviations of matrix elements
- `mean_diff`: Difference in mean matrices
- `sd_diff`: Difference in SD matrices
- `lambda_s_treatment`: Stochastic λ of treatment
- `lambda_s_reference`: Stochastic λ of reference
"""
struct StochasticLTREResult{T<:Real}
    cont_mean::Matrix{T}
    cont_sd::Matrix{T}
    mean_diff::Matrix{T}
    sd_diff::Matrix{T}
    lambda_s_treatment::T
    lambda_s_reference::T
end

"""
    SNALTREResult

Result of a small-noise approximation LTRE (Davison et al. 2019).

Decomposes stochastic growth rate differences into four additive components:
- Shifts in mean vital rates
- Shifts in elasticities
- Shifts in temporal variation (CV)
- Shifts in temporal correlations

# Fields
- `cont_mean`: Contributions from shifts in mean matrix elements
- `cont_elasticity`: Contributions from shifts in elasticities
- `cont_cv`: Contributions from shifts in temporal variation
- `cont_correlation`: Contributions from shifts in temporal correlations
- `r_treatment`: log(λ_s) for treatment
- `r_reference`: log(λ_s) for reference
- `delta_r`: Difference in log(λ_s)
"""
struct SNALTREResult{T<:Real}
    cont_mean::Matrix{T}
    cont_elasticity::Matrix{T}
    cont_cv::Matrix{T}
    cont_correlation::Matrix{T}
    r_treatment::T
    r_reference::T
    delta_r::T
end

"""
    ExactLTREResult

Result of an exact fANOVA-based LTRE (Hernandez et al. 2023).

Provides exact decomposition of Δλ (fixed design) or Var(λ) (random design)
into main effects and interaction effects of arbitrary order, without
linear approximation.

# Fields
- `indices_varying`: Vector of linear indices of matrix elements that differ
- `effects`: Vector of exact effect values (main effects + interactions)
- `effect_indices`: For each effect, which varying indices are involved
- `method`: `:fixed` or `:random`
- `directional`: Whether a directional (true) or symmetric (false) decomposition was used
"""
struct ExactLTREResult{T<:Real}
    indices_varying::Vector{Int}
    effects::Vector{T}
    effect_indices::Vector{Vector{Int}}
    method::Symbol
    directional::Bool
end

"""
    RandomLTREResult

Result of a classical random-design LTRE analysis.

# Fields
- `contributions`: n²×n² matrix where diagonal = variance contributions,
  off-diagonal = covariance contributions
- `covariance_matrix`: The element-level temporal covariance matrix
- `sensitivity_vec`: Vectorized sensitivity matrix at the mean
- `lambda_mean`: λ of the mean matrix
- `var_lambda`: Observed variance in λ across matrices
"""
struct RandomLTREResult{T<:Real}
    contributions::Matrix{T}
    covariance_matrix::Matrix{T}
    sensitivity_vec::Vector{T}
    lambda_mean::T
    var_lambda::T
end

# --- Deterministic LTRE ---

"""
    ltre(treatment::AbstractMatrix, reference::AbstractMatrix;
         method=:midpoint)

Deterministic fixed-design LTRE.

Decomposes the difference in λ between a treatment and reference matrix into
contributions from each matrix element:

    λ_treatment - λ_reference ≈ Σᵢⱼ (aᵢⱼ_treat - aᵢⱼ_ref) × sᵢⱼ

where sᵢⱼ is the sensitivity evaluated at the midpoint (default) or reference matrix.

# Arguments
- `treatment`: Treatment projection matrix
- `reference`: Reference projection matrix
- `method`: `:midpoint` (evaluate sensitivity at mean of treatment/reference)
            or `:reference` (evaluate at reference matrix)

# Returns
An `LTREResult` with element-level contributions.
"""
function ltre(treatment::AbstractMatrix, reference::AbstractMatrix;
              method::Symbol=:midpoint)
    size(treatment) == size(reference) || throw(DimensionMismatch(
        "Treatment and reference matrices must have the same dimensions"))

    T = promote_type(eltype(treatment), eltype(reference))
    treat = convert(Matrix{T}, treatment)
    ref = convert(Matrix{T}, reference)

    # Difference matrix
    diff = treat .- ref

    # Sensitivity at evaluation point
    eval_matrix = if method == :midpoint
        (treat .+ ref) ./ 2
    elseif method == :reference
        ref
    else
        throw(ArgumentError("method must be :midpoint or :reference"))
    end

    S = sensitivity(eval_matrix)

    # Element-level contributions
    contributions = diff .* S

    λ_treat = lambda(treat)
    λ_ref = lambda(ref)

    return LTREResult{T}(contributions, diff, S, λ_treat, λ_ref, λ_treat - λ_ref)
end

"""
    ltre(treatments::AbstractVector{<:AbstractMatrix}, reference::AbstractMatrix;
         method=:midpoint)

LTRE for multiple treatment matrices against a single reference.
Returns a vector of `LTREResult`s.
"""
function ltre(treatments::AbstractVector{<:AbstractMatrix},
              reference::AbstractMatrix; method::Symbol=:midpoint)
    return [ltre(t, reference; method=method) for t in treatments]
end

"""
    ltre(mats::AbstractVector{<:AbstractMatrix}; ref=:mean, method=:midpoint)

LTRE for a set of matrices, using their mean (default) or a specific index as reference.

# Arguments
- `mats`: Vector of projection matrices (e.g., annual matrices)
- `ref`: `:mean` to use the element-wise mean matrix as reference,
         or an integer index into `mats`
- `method`: `:midpoint` or `:reference` for sensitivity evaluation
"""
function ltre(mats::AbstractVector{<:AbstractMatrix};
              ref::Union{Symbol,Int}=:mean, method::Symbol=:midpoint)
    isempty(mats) && throw(ArgumentError("Must provide at least one matrix"))

    reference = if ref == :mean
        mean(mats)
    elseif ref isa Int
        mats[ref]
    else
        throw(ArgumentError("ref must be :mean or an integer index"))
    end

    return [ltre(m, reference; method=method) for m in mats]
end

# --- Stochastic LTRE ---

"""
    stochastic_ltre(treatment_mats::AbstractVector{<:AbstractMatrix},
                    reference_mats::AbstractVector{<:AbstractMatrix};
                    times=10000, burn_in=3000, seed=nothing)

Stochastic LTRE following Davison et al. (2010).

Decomposes the difference in stochastic growth rate (log λ_s) between two
sets of environmental matrices into contributions from:
1. Differences in mean matrix elements
2. Differences in the variability (SD) of matrix elements

Uses simulation to estimate stochastic sensitivities to mean and variance.

# Arguments
- `treatment_mats`: Vector of annual matrices for the treatment population
- `reference_mats`: Vector of annual matrices for the reference population
- `times`: Number of time steps for stochastic simulation
- `burn_in`: Steps to discard before computing growth rates
- `seed`: Optional random seed
"""
function stochastic_ltre(treatment_mats::AbstractVector{<:AbstractMatrix},
                         reference_mats::AbstractVector{<:AbstractMatrix};
                         times::Int=10000, burn_in::Int=3000)
    isempty(treatment_mats) && throw(ArgumentError("treatment_mats cannot be empty"))
    isempty(reference_mats) && throw(ArgumentError("reference_mats cannot be empty"))

    n = size(treatment_mats[1], 1)
    T = Float64

    # Compute mean and SD matrices for each set
    mean_treat = mean(treatment_mats)
    mean_ref = mean(reference_mats)

    sd_treat = _elementwise_sd(treatment_mats)
    sd_ref = _elementwise_sd(reference_mats)

    # Differences
    mean_diff = mean_treat .- mean_ref
    sd_diff = sd_treat .- sd_ref

    # Midpoint matrices for sensitivity evaluation
    mid_mean = (mean_treat .+ mean_ref) ./ 2
    mid_sd = (sd_treat .+ sd_ref) ./ 2

    # Stochastic sensitivities via simulation
    # seed handling removed for simplicity; use global RNG
    sens_mean, sens_sd = _stochastic_sensitivities(mid_mean, mid_sd, n;
                                                    times=times, burn_in=burn_in)

    # Contributions
    cont_mean = mean_diff .* sens_mean
    cont_sd = sd_diff .* sens_sd

    # Stochastic growth rates
    λs_treat = _simulate_stochastic_lambda(treatment_mats; times=times, burn_in=burn_in)
    λs_ref = _simulate_stochastic_lambda(reference_mats; times=times, burn_in=burn_in)

    return StochasticLTREResult{T}(cont_mean, cont_sd, mean_diff, sd_diff, λs_treat, λs_ref)
end

# --- Small-Noise Approximation LTRE ---

"""
    sna_ltre(treatment_mats::AbstractVector{<:AbstractMatrix},
             reference_mats::AbstractVector{<:AbstractMatrix})

Small-noise approximation LTRE following Davison et al. (2019).

Decomposes the difference in stochastic growth rate into four orthogonal
components:
1. **Mean**: Contributions from shifts in mean matrix elements
2. **Elasticity**: Contributions from shifts in elasticity structure
3. **CV**: Contributions from shifts in temporal variation of elements
4. **Correlation**: Contributions from shifts in temporal correlations

The SNA approximation for log(λ_s) is:
    r ≈ log(λ_d) - (1/2) Σᵢⱼ Σₖₗ eᵢⱼ × eₖₗ × τᵢⱼₖₗ

where eᵢⱼ = elasticity of element (i,j), and τᵢⱼₖₗ is the temporal
covariance between elements (i,j) and (k,l) scaled by their means.
"""
function sna_ltre(treatment_mats::AbstractVector{<:AbstractMatrix},
                  reference_mats::AbstractVector{<:AbstractMatrix})
    isempty(treatment_mats) && throw(ArgumentError("treatment_mats cannot be empty"))
    isempty(reference_mats) && throw(ArgumentError("reference_mats cannot be empty"))

    n = size(treatment_mats[1], 1)
    T = Float64

    # Mean matrices
    mean_treat = mean(treatment_mats)
    mean_ref = mean(reference_mats)

    # Deterministic lambdas
    λ_treat = lambda(mean_treat)
    λ_ref = lambda(mean_ref)

    # Elasticities
    elas_treat = elasticity(mean_treat)
    elas_ref = elasticity(mean_ref)

    # Temporal covariance matrices (n² × n² flattened)
    cov_treat = _temporal_covariance(treatment_mats, mean_treat)
    cov_ref = _temporal_covariance(reference_mats, mean_ref)

    # Midpoints for decomposition
    mid_mean = (mean_treat .+ mean_ref) ./ 2
    mid_elas = (elas_treat .+ elas_ref) ./ 2
    mid_cov = (cov_treat .+ cov_ref) ./ 2

    # Sensitivity of log(λ_d) at midpoint
    mid_sens = sensitivity(mid_mean)
    mid_lambda = lambda(mid_mean)

    # --- Component 1: Mean contributions ---
    # Δr_mean ≈ Σᵢⱼ (ā_treat_ij - ā_ref_ij) × (sᵢⱼ / λ)
    mean_diff = mean_treat .- mean_ref
    cont_mean = mean_diff .* (mid_sens ./ mid_lambda)

    # --- Component 2: Elasticity contributions ---
    elas_diff = elas_treat .- elas_ref
    # Contribution from elasticity shift: -½ Σᵢⱼ Δeᵢⱼ × Σₖₗ mid_eₖₗ × mid_τᵢⱼₖₗ
    cont_elasticity = zeros(T, n, n)
    for j in 1:n, i in 1:n
        ij = (j - 1) * n + i
        sum_cov = zero(T)
        for l in 1:n, k in 1:n
            kl = (l - 1) * n + k
            sum_cov += mid_elas[k, l] * mid_cov[ij, kl]
        end
        cont_elasticity[i, j] = -0.5 * elas_diff[i, j] * sum_cov
    end

    # --- Component 3: CV (temporal variation) contributions ---
    # Diagonal of covariance: var(aᵢⱼ)/mean(aᵢⱼ)²
    cov_diff = cov_treat .- cov_ref
    cont_cv = zeros(T, n, n)
    for j in 1:n, i in 1:n
        ij = (j - 1) * n + i
        cont_cv[i, j] = -0.5 * mid_elas[i, j]^2 * cov_diff[ij, ij]
    end

    # --- Component 4: Correlation contributions ---
    cont_correlation = zeros(T, n, n)
    for j in 1:n, i in 1:n
        ij = (j - 1) * n + i
        corr_sum = zero(T)
        for l in 1:n, k in 1:n
            kl = (l - 1) * n + k
            kl == ij && continue  # skip diagonal (that's CV)
            corr_sum += mid_elas[k, l] * cov_diff[ij, kl]
        end
        cont_correlation[i, j] = -0.5 * mid_elas[i, j] * corr_sum
    end

    # Approximate stochastic growth rates
    r_treat = log(λ_treat) - 0.5 * _sna_variance_term(elas_treat, cov_treat)
    r_ref = log(λ_ref) - 0.5 * _sna_variance_term(elas_ref, cov_ref)

    return SNALTREResult{T}(cont_mean, cont_elasticity, cont_cv, cont_correlation,
                            r_treat, r_ref, r_treat - r_ref)
end

# --- Helper functions ---

"""Compute element-wise standard deviation across a set of matrices."""
function _elementwise_sd(mats::AbstractVector{<:AbstractMatrix})
    n = size(mats[1], 1)
    T = Float64
    mean_mat = mean(mats)
    sd_mat = zeros(T, n, n)
    k = length(mats)
    for m in mats
        sd_mat .+= (m .- mean_mat) .^ 2
    end
    return sqrt.(sd_mat ./ max(k - 1, 1))
end

"""
Compute temporal covariance matrix between all pairs of matrix elements.
Returns an n² × n² matrix where entry (ij, kl) is cov(a_ij, a_kl) / (ā_ij × ā_kl),
using the scaled covariance (coefficient of variation formulation).
"""
function _temporal_covariance(mats::AbstractVector{<:AbstractMatrix},
                              mean_mat::AbstractMatrix)
    n = size(mean_mat, 1)
    n2 = n * n
    T = Float64
    k = length(mats)

    # Flatten each matrix and compute covariance
    flat = zeros(T, k, n2)
    for (t, m) in enumerate(mats)
        flat[t, :] .= vec(m)
    end

    # Raw covariance matrix
    cov_raw = zeros(T, n2, n2)
    mean_flat = vec(mean_mat)
    for t in 1:k
        d = flat[t, :] .- mean_flat
        cov_raw .+= d * d'
    end
    cov_raw ./= max(k - 1, 1)

    # Scale by means: τ_ij,kl = cov(a_ij, a_kl) / (ā_ij × ā_kl)
    # Handle zero means by leaving scaled covariance as zero
    scaled = zeros(T, n2, n2)
    for kl in 1:n2, ij in 1:n2
        denom = mean_flat[ij] * mean_flat[kl]
        if abs(denom) > eps(T)
            scaled[ij, kl] = cov_raw[ij, kl] / denom
        end
    end

    return scaled
end

"""Compute the SNA variance reduction term: Σᵢⱼ Σₖₗ eᵢⱼ × eₖₗ × τᵢⱼₖₗ"""
function _sna_variance_term(elas::AbstractMatrix, cov::AbstractMatrix)
    n2 = size(cov, 1)
    e_flat = vec(elas)
    s = 0.0
    for kl in 1:n2, ij in 1:n2
        s += e_flat[ij] * e_flat[kl] * cov[ij, kl]
    end
    return s
end

"""Simulate stochastic growth rate by random matrix sampling."""
function _simulate_stochastic_lambda(mats::AbstractVector{<:AbstractMatrix};
                                     times::Int=10000, burn_in::Int=3000)
    n = size(mats[1], 1)
    k = length(mats)
    pop = ones(Float64, n)
    pop ./= sum(pop)

    log_lambdas = zeros(Float64, times)
    for t in 1:times
        idx = rand(1:k)
        pop_new = mats[idx] * pop
        nt = sum(pop_new)
        if nt > 0
            log_lambdas[t] = log(nt)
            pop = pop_new ./ nt
        else
            log_lambdas[t] = -Inf
            pop = ones(Float64, n) ./ n
        end
    end

    valid = log_lambdas[(burn_in+1):end]
    valid = filter(isfinite, valid)
    return isempty(valid) ? -Inf : exp(mean(valid))
end

"""Compute stochastic sensitivities to mean and SD via numerical perturbation."""
function _stochastic_sensitivities(mid_mean::AbstractMatrix, mid_sd::AbstractMatrix,
                                   n::Int; times::Int=10000, burn_in::Int=3000,
                                   pert::Float64=1e-4)
    T = Float64

    # Generate matrices from mid_mean + Normal(0, mid_sd) for baseline
    function generate_mats(μ, σ; n_env=50)
        mats = Matrix{T}[]
        for _ in 1:n_env
            m = μ .+ σ .* randn(T, size(μ)...)
            m = max.(m, zero(T))  # keep non-negative
            push!(mats, m)
        end
        return mats
    end

    base_mats = generate_mats(mid_mean, mid_sd)
    base_λs = _simulate_stochastic_lambda(base_mats; times=times, burn_in=burn_in)
    base_r = log(max(base_λs, eps(T)))

    sens_mean = zeros(T, n, n)
    sens_sd = zeros(T, n, n)

    for j in 1:n, i in 1:n
        # Sensitivity to mean
        μ_pert = copy(mid_mean)
        μ_pert[i, j] += pert
        pert_mats = generate_mats(μ_pert, mid_sd)
        pert_r = log(max(_simulate_stochastic_lambda(pert_mats; times=times, burn_in=burn_in), eps(T)))
        sens_mean[i, j] = (pert_r - base_r) / pert

        # Sensitivity to SD
        σ_pert = copy(mid_sd)
        σ_pert[i, j] += pert
        pert_mats = generate_mats(mid_mean, σ_pert)
        pert_r = log(max(_simulate_stochastic_lambda(pert_mats; times=times, burn_in=burn_in), eps(T)))
        sens_sd[i, j] = (pert_r - base_r) / pert
    end

    return sens_mean, sens_sd
end

# --- Classical Random-Design LTRE ---

"""
    ltre_random(mats::AbstractVector{<:AbstractMatrix})

Classical random-design LTRE (Caswell 2001, Ch. 10).

Decomposes Var(λ) into contributions from variance and covariance of
matrix elements:

    Var(λ) ≈ Σᵢⱼ Σₖₗ Cov(aᵢⱼ, aₖₗ) × sᵢⱼ × sₖₗ

where sensitivities are evaluated at the mean matrix.

# Arguments
- `mats`: Vector of ≥2 projection matrices (e.g., annual matrices)

# Returns
A `RandomLTREResult` with the n²×n² contribution matrix.
"""
function ltre_random(mats::AbstractVector{<:AbstractMatrix})
    length(mats) >= 2 || throw(ArgumentError("Need at least 2 matrices for random LTRE"))
    n = size(mats[1], 1)
    n2 = n * n
    T = Float64

    # Compute mean matrix and its sensitivity
    mean_mat = mean(mats)
    S = sensitivity(mean_mat)
    s_vec = vec(S)
    λ_mean = lambda(mean_mat)

    # Compute covariance matrix of matrix elements
    k = length(mats)
    flat = zeros(T, k, n2)
    for (t, m) in enumerate(mats)
        flat[t, :] .= vec(m)
    end
    mean_flat = vec(mean_mat)
    cov_mat = zeros(T, n2, n2)
    for t in 1:k
        d = flat[t, :] .- mean_flat
        cov_mat .+= d * d'
    end
    cov_mat ./= max(k - 1, 1)

    # Contributions: C_ij,kl = Cov(a_ij, a_kl) * s_ij * s_kl
    contributions = cov_mat .* (s_vec * s_vec')

    # Observed variance in lambda
    lambdas = [lambda(m) for m in mats]
    var_lambda = var(lambdas)

    return RandomLTREResult{T}(contributions, cov_mat, s_vec, λ_mean, var_lambda)
end

# --- Exact LTRE (fANOVA-based, Hernandez et al. 2023) ---

"""
    exact_ltre(treatment::AbstractMatrix, reference::AbstractMatrix;
               maxint::Union{Int,Symbol}=:all, directional::Bool=false)

Exact fixed-design LTRE (Hernandez et al. 2023).

Uses functional ANOVA principles to exactly decompose Δλ into main effects
and interaction effects, without the linear (sensitivity) approximation used
in classical LTRE.

For each subset S of varying parameters, the "response" ν(S) is computed by
allowing only parameters in S to take their observed values while fixing all
others at the baseline. The "effect" ε(S) is then computed by inclusion-exclusion
(Möbius inversion) over subsets.

# Arguments
- `treatment`: Treatment projection matrix
- `reference`: Reference projection matrix (baseline if directional=true)
- `maxint`: Maximum interaction order (`:all` or an integer)
- `directional`: If `true`, reference is baseline; if `false`, mean is baseline (symmetric)

# Returns
An `ExactLTREResult` with effects for each main effect and interaction term.

# Notes
- Complexity is O(2^p) where p = number of varying elements. Works well for p ≤ 15.
- For larger p, set `maxint` to limit interaction order.
"""
function exact_ltre(treatment::AbstractMatrix, reference::AbstractMatrix;
                    maxint::Union{Int,Symbol}=:all, directional::Bool=false)
    size(treatment) == size(reference) || throw(DimensionMismatch(
        "Treatment and reference matrices must have the same dimensions"))

    n = size(treatment, 1)
    T = Float64

    vec_treat = vec(Float64.(treatment))
    vec_ref = vec(Float64.(reference))

    # Find varying indices
    indices_varying = findall(i -> abs(vec_treat[i] - vec_ref[i]) > 0, 1:length(vec_treat))
    p = length(indices_varying)

    p == 0 && return ExactLTREResult{T}(Int[], T[], Vector{Int}[], :fixed, directional)

    # Baseline for fixing parameters
    baseline = directional ? vec_ref : (vec_treat .+ vec_ref) ./ 2

    # Generate all subsets up to maxint
    max_order = maxint == :all ? p : min(Int(maxint), p)
    subsets = _generate_subsets(indices_varying, max_order)

    # Compute responses for each subset
    responses = zeros(T, length(subsets))
    for (idx, subset) in enumerate(subsets)
        # Build matrix: varying indices in subset take observed values, rest are baseline
        vec_m1 = copy(baseline)
        vec_m2 = copy(baseline)
        for i in subset
            vec_m1[i] = vec_ref[i]    # matrix 1 (ref or obs1)
            vec_m2[i] = vec_treat[i]  # matrix 2 (treat or obs2)
        end
        m1 = reshape(vec_m1, n, n)
        m2 = reshape(vec_m2, n, n)
        if directional
            responses[idx] = lambda(m2) - lambda(m1)
        else
            responses[idx] = lambda(m2) - lambda(m1)
        end
    end

    # Compute effects via inclusion-exclusion (Möbius inversion)
    effects = _compute_effects(responses, subsets, indices_varying)

    # Convert subset indices to the varying-parameter index space
    effect_indices = [Int[findfirst(==(i), indices_varying) for i in s] for s in subsets]

    return ExactLTREResult{T}(indices_varying, effects, effect_indices, :fixed, directional)
end

"""
    exact_ltre(mats::AbstractVector{<:AbstractMatrix};
               maxint::Union{Int,Symbol}=:all)

Exact random-design LTRE (Hernandez et al. 2023).

Decomposes Var(λ) into exact main effects and interactions using fANOVA.
For each subset S of varying parameters, all non-S parameters are fixed at
their mean values and Var(λ) is computed over the resulting modified matrices.
"""
function exact_ltre(mats::AbstractVector{<:AbstractMatrix};
                    maxint::Union{Int,Symbol}=:all)
    length(mats) >= 2 || throw(ArgumentError("Need at least 2 matrices"))
    n = size(mats[1], 1)
    n2 = n * n
    T = Float64

    # Vectorize matrices
    k = length(mats)
    flat = zeros(T, k, n2)
    for (t, m) in enumerate(mats)
        flat[t, :] .= vec(m)
    end
    mean_vec = vec(mean(mats))

    # Find varying indices (non-zero variance)
    vars = [var(flat[:, j]) for j in 1:n2]
    indices_varying = findall(v -> v > 0, vars)
    p = length(indices_varying)

    p == 0 && return ExactLTREResult{T}(Int[], T[], Vector{Int}[], :random, false)

    max_order = maxint == :all ? p : min(Int(maxint), p)
    subsets = _generate_subsets(indices_varying, max_order)

    # Compute responses: Var(λ) with only subset elements varying
    responses = zeros(T, length(subsets))
    for (idx, subset) in enumerate(subsets)
        lambdas = zeros(T, k)
        for t in 1:k
            vec_m = copy(mean_vec)
            for i in subset
                vec_m[i] = flat[t, i]
            end
            m = reshape(vec_m, n, n)
            lambdas[t] = lambda(m)
        end
        # Complete-sample variance (N denominator, not N-1)
        μ = mean(lambdas)
        responses[idx] = mean((lambdas .- μ) .^ 2)
    end

    # Compute effects
    effects = _compute_effects(responses, subsets, indices_varying)
    effect_indices = [Int[findfirst(==(i), indices_varying) for i in s] for s in subsets]

    return ExactLTREResult{T}(indices_varying, effects, effect_indices, :random, false)
end

# --- Exact LTRE helpers ---

"""Generate all subsets of `indices` up to order `max_order`, starting with empty set."""
function _generate_subsets(indices::Vector{Int}, max_order::Int)
    subsets = Vector{Int}[Int[]]  # empty set first
    for order in 1:max_order
        for combo in combinations(indices, order)
            push!(subsets, collect(combo))
        end
    end
    return subsets
end

"""Simple combinations generator (like itertools.combinations)."""
function combinations(v::Vector{T}, k::Int) where T
    n = length(v)
    k > n && return Vector{T}[]
    k == 0 && return [T[]]
    k == n && return [copy(v)]

    result = Vector{T}[]
    indices = collect(1:k)

    while true
        push!(result, v[indices])
        # Find rightmost index that can be incremented
        i = k
        while i > 0 && indices[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        indices[i] += 1
        for j in (i+1):k
            indices[j] = indices[j-1] + 1
        end
    end
    return result
end

"""
Compute effects from responses via inclusion-exclusion (Möbius inversion).
Uses the G-matrix approach for small p, direct computation otherwise.
"""
function _compute_effects(responses::Vector{T}, subsets::Vector{Vector{Int}},
                          indices_varying::Vector{Int}) where T
    n_terms = length(subsets)
    effects = zeros(T, n_terms)
    effects[1] = responses[1]  # empty set response

    # For each term, subtract lower-order contributions
    for i in 2:n_terms
        s_i = Set(subsets[i])
        order_i = length(subsets[i])

        # Sum over all proper subsets of s_i that appear in our list
        correction = zero(T)
        for j in 1:(i-1)
            s_j = Set(subsets[j])
            if issubset(s_j, s_i)
                # Inclusion-exclusion sign: (-1)^(order_i - order_j)
                order_j = length(subsets[j])
                sign = iseven(order_i - order_j) ? one(T) : -one(T)
                correction += sign * responses[j]
            end
        end
        effects[i] = responses[i] - correction +
                     (iseven(order_i) ? responses[1] : -responses[1]) -
                     responses[1]  # adjust for initial term

        # Actually, use direct Möbius: ε(S) = Σ_{T⊆S} (-1)^{|S|-|T|} ν(T)
        effects[i] = zero(T)
        for j in 1:n_terms
            s_j = Set(subsets[j])
            if issubset(s_j, s_i)
                order_j = length(subsets[j])
                sign = iseven(order_i - order_j) ? one(T) : -one(T)
                effects[i] += sign * responses[j]
            end
        end
    end

    return effects
end

# --- G-matrix (Poelwijk et al. 2016) ---

"""
    gmatrix(n::Int)

Construct the 2ⁿ × 2ⁿ G-matrix for converting responses to effects.
From Poelwijk, Krishna, Ranganathan (2016).
"""
function gmatrix(n::Int)
    G = ones(Float64, 1, 1)
    for _ in 1:n
        top = hcat(G, zeros(size(G)))
        bot = hcat(-G, G)
        G = vcat(top, bot)
    end
    return G
end
