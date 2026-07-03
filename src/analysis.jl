"""
Analysis functions operating on matrices and AbstractProjectionSolution.
"""

# --- Matrix-level dispatches ---

"""
    lambda(A::AbstractMatrix)

Dominant eigenvalue of projection matrix A.
"""
function lambda(A::AbstractMatrix)
    F = eigen(A)
    return maximum(real.(F.values))
end

"""
    stable_distribution(A::AbstractMatrix)

Normalized right eigenvector of A (stable stage/size distribution).
"""
function stable_distribution(A::AbstractMatrix)
    ea = eigenanalysis_full(A)
    return ea.stable_dist
end

"""
    reproductive_value(A::AbstractMatrix)

Normalized left eigenvector of A (reproductive value).
"""
function reproductive_value(A::AbstractMatrix)
    ea = eigenanalysis_full(A)
    return ea.repro_value
end

"""
    sensitivity(A::AbstractMatrix)

Sensitivity matrix: S[i,j] = v[i] * w[j] / dot(v, w)
where v = reproductive value, w = stable distribution.
With dot(v,w)=1 normalization, this simplifies to v * w'.
"""
function sensitivity(A::AbstractMatrix)
    ea = eigenanalysis_full(A)
    v = ea.repro_value
    w = ea.stable_dist
    return v * w'
end

"""
    elasticity(A::AbstractMatrix)

Elasticity matrix: E[i,j] = (A[i,j] / lambda) * S[i,j]
Entries sum to 1.0.
"""
function elasticity(A::AbstractMatrix)
    ea = eigenanalysis_full(A)
    S = ea.repro_value * ea.stable_dist'
    return (A .* S) ./ ea.lambda
end

"""
    damping_ratio(A::AbstractMatrix)

Ratio of dominant to subdominant eigenvalue magnitudes.
Measures rate of convergence to stable distribution.
"""
function damping_ratio(A::AbstractMatrix)
    F = eigen(A)
    vals = sort(abs.(F.values), rev=true)
    length(vals) < 2 && return Inf
    abs(vals[2]) < eps(Float64) && return Inf
    return vals[1] / vals[2]
end

# --- Solution-level dispatches ---

"""
    lambda(sol::AbstractProjectionSolution; burn_in_frac=0.1)

Dominant eigenvalue from solution. Uses eigenanalysis if available,
otherwise computes stochastic growth rate from per-step lambdas.
"""
function lambda(sol::AbstractProjectionSolution; burn_in_frac=0.1)
    if sol.eigenanalysis !== nothing
        return sol.eigenanalysis.lambda
    end
    if !isempty(sol.lambdas)
        burn_in = round(Int, length(sol.lambdas) * burn_in_frac)
        return stochastic_growth_rate(sol; burn_in=burn_in)
    end
    error("No lambda available. Run with DirectIteration or EigenAnalysis.")
end

"""
    stable_distribution(sol::AbstractProjectionSolution)

Stable distribution from solution eigenanalysis, or approximated from final state.
"""
function stable_distribution(sol::AbstractProjectionSolution)
    if sol.eigenanalysis !== nothing
        return sol.eigenanalysis.stable_dist
    end
    n = sol.u[end]
    s = sum(n)
    return s > 0 ? n ./ s : n
end

"""
    reproductive_value(sol::AbstractProjectionSolution)

Reproductive value from solution eigenanalysis, or computed from kernel matrix.
"""
function reproductive_value(sol::AbstractProjectionSolution)
    if sol.eigenanalysis !== nothing
        return sol.eigenanalysis.repro_value
    end
    K = sol.kernel_matrices
    if K isa AbstractMatrix
        return reproductive_value(K)
    end
    error("Reproductive value requires eigenanalysis or a single kernel matrix.")
end

"""
    sensitivity(sol::AbstractProjectionSolution)

Sensitivity matrix from solution.
"""
function sensitivity(sol::AbstractProjectionSolution)
    K = _get_single_kernel(sol)
    return sensitivity(K)
end

"""
    elasticity(sol::AbstractProjectionSolution)

Elasticity matrix from solution.
"""
function elasticity(sol::AbstractProjectionSolution)
    K = _get_single_kernel(sol)
    return elasticity(K)
end

"""
    stochastic_growth_rate(sol::AbstractProjectionSolution; burn_in=0)

Stochastic growth rate: `exp(mean(log(lambda_t)))` for `t > burn_in`.
Geometric mean of per-step growth rates on the log scale.
"""
function stochastic_growth_rate(sol::AbstractProjectionSolution; burn_in=0)
    isempty(sol.lambdas) && error("No per-step lambdas available.")
    λs = sol.lambdas[(burn_in + 1):end]
    pos = filter(x -> x > 0, λs)
    isempty(pos) && error("All per-step lambdas are non-positive")
    length(pos) < length(λs) && @warn "Some per-step lambdas were non-positive and excluded"
    return exp(mean(log.(pos)))
end

"""
    mean_kernel(sol::AbstractProjectionSolution)

Mean kernel matrix from a stochastic simulation.
Returns the single matrix for deterministic solves.
"""
function mean_kernel(sol::AbstractProjectionSolution)
    K = sol.kernel_matrices
    if K isa AbstractVector
        return mean(K)
    end
    return K
end

# --- Helper ---

function _get_single_kernel(sol::AbstractProjectionSolution)
    K = sol.kernel_matrices
    if K isa AbstractMatrix
        return K
    elseif K isa AbstractVector && !isempty(K)
        return mean(K)
    end
    error("No kernel matrix available.")
end
