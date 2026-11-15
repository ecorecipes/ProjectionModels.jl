"""
Operations between kernels, matrices and block matrices that the model packages share.

- **Left Kan extension**: kernel function → matrix (discretisation)
- **Right Kan extension**: matrix → piecewise-constant kernel (refinement)
- **Stratification**: local matrix + coupling → block-structured matrix
- **Coarsening**: fine matrix → coarse matrix by aggregating bins

The methods here act on the core types (`ContinuousDomain`, matrices) and need no other
package. IntegralProjectionModels.jl re-exports them. CategoricalPopulationDynamics.jl
extends the same functions to its own domains, to projection nets and to Catlab's
`FinFunction`. Because every package extends these functions rather than defining its own
of the same name, the names can be used unqualified whichever packages are loaded.
"""

"""
    left_kan_extension(kernel_fn, domain::ContinuousDomain)

Discretise a continuous kernel function `kernel_fn(z_new, z)` into a matrix
using the midpoint rule on the given domain.

This is the **left Kan extension** along the discretisation functor:
it maps a morphism in **Krn** (Markov kernels on measurable spaces) to a
morphism in **FinStoch** (stochastic matrices on finite sets).

``A_{ij} = h \\cdot K(z_i, z_j)``

where ``h`` is the bin width and ``z_i`` are the midpoints.

# Arguments
- `kernel_fn`: a callable `(z_new, z) -> Real` giving the kernel density
- `domain`: a `ContinuousDomain` specifying bounds and mesh resolution

# Returns
- `A::Matrix{Float64}`: the discretised transition matrix

Keyword options (another quadrature rule, positivity) are implemented by
CategoricalPopulationDynamics.jl; see [`discretise_kernel`](@ref).
"""
function left_kan_extension(kernel_fn, domain::ContinuousDomain; options...)
    isempty(options) || return discretise_kernel(kernel_fn, domain; options...)
    z = meshpoints(domain)
    h = step_size(domain)
    m = length(z)
    A = zeros(m, m)
    for j in 1:m
        for i in 1:m
            A[i, j] = h * kernel_fn(z[i], z[j])
        end
    end
    return A
end

"""
    discretise_kernel(kernel_fn, domain; options...)

Extension point for [`left_kan_extension`](@ref) called with keyword options on a
`ContinuousDomain`. The core package implements the midpoint rule only.
CategoricalPopulationDynamics.jl adds the method that honours `rule`, `ensure_nonneg` and
`normalize_rows`. Not exported.
"""
function discretise_kernel(kernel_fn, domain; options...)
    throw(ArgumentError(
        "left_kan_extension on a ContinuousDomain uses the midpoint rule; the options " *
        "$(join(keys(options), ", ")) need CategoricalPopulationDynamics.jl to be loaded"))
end

"""
    right_kan_extension(A::AbstractMatrix, domain::ContinuousDomain)

Construct a piecewise-constant kernel function from a matrix and domain.

This is the **right Kan extension**: the most conservative continuous
representation of a discrete transition matrix. The resulting kernel is
constant within each bin pair:

``K_{\\mathrm{pw}}(z', z) = A_{ij} / h \\quad \\text{where } z \\in B_j,\\; z' \\in B_i``

# Arguments
- `A`: an `n × n` transition matrix
- `domain`: a `ContinuousDomain` with `n` meshpoints matching `size(A, 1)`

# Returns
- A callable `(z_new, z) -> Real` giving the piecewise-constant kernel density
"""
function right_kan_extension(A::AbstractMatrix, domain::ContinuousDomain)
    z_min = domain.lower
    h = step_size(domain)
    n = size(A, 1)
    function piecewise_kernel(z_new, z)
        j = clamp(Int(ceil((z - z_min) / h)), 1, n)
        i = clamp(Int(ceil((z_new - z_min) / h)), 1, n)
        return A[i, j] / h
    end
    return piecewise_kernel
end

"""
    stratify(A_local::AbstractMatrix, coupling::AbstractMatrix)

Construct a block-structured matrix from a single local transition matrix and a
coupling (dispersal) matrix. Every stratum (patch, genotype, infection state) shares
the same local dynamics.

In the categorical framework, this is a **pullback** in a slice category:
the local demography is replicated across strata and weighted by the coupling.

``A_{\\mathrm{strat}}[(s_{\\mathrm{to}}, i),\\, (s_{\\mathrm{from}}, j)]
  = D[s_{\\mathrm{to}}, s_{\\mathrm{from}}] \\cdot A_{\\mathrm{local}}[i, j]``

# Arguments
- `A_local`: an `n × n` local (single-stratum) transition matrix
- `coupling`: a `p × p` matrix of weights between strata, `coupling[to, from]`

# Returns
- an `(n·p) × (n·p)` block-structured matrix
"""
function stratify(A_local::AbstractMatrix, coupling::AbstractMatrix)
    size(A_local, 1) == size(A_local, 2) ||
        throw(DimensionMismatch("local matrix must be square, got $(size(A_local))"))
    size(coupling, 1) == size(coupling, 2) ||
        throw(DimensionMismatch("coupling matrix must be square, got $(size(coupling))"))
    n_bins = size(A_local, 1)
    n_strata = size(coupling, 1)
    n_total = n_bins * n_strata
    T = promote_type(eltype(A_local), eltype(coupling))
    A_strat = zeros(T, n_total, n_total)
    for s_to in 1:n_strata
        for s_from in 1:n_strata
            rows = ((s_to - 1) * n_bins + 1):(s_to * n_bins)
            cols = ((s_from - 1) * n_bins + 1):(s_from * n_bins)
            A_strat[rows, cols] .= coupling[s_to, s_from] .* A_local
        end
    end
    return A_strat
end

"""
    coarsen(A::AbstractMatrix, from_domain::ContinuousDomain, to_domain::ContinuousDomain)

Coarsen a fine transition matrix to a coarser resolution by aggregating bins.

This is the **pushforward** (left Kan extension) along the coarsening map
``f\\colon \\mathrm{FinSet}(n_{\\mathrm{fine}}) \\to \\mathrm{FinSet}(n_{\\mathrm{coarse}})``.
Each coarse bin collects contributions from the fine bins it contains.

The `from_domain` and `to_domain` must share the same `[lower, upper]` bounds.
The number of fine bins must be an integer multiple of the number of coarse bins.

# Arguments
- `A`: an `n_fine × n_fine` transition matrix
- `from_domain`: the fine `ContinuousDomain`
- `to_domain`: the coarse `ContinuousDomain`

# Returns
- an `n_coarse × n_coarse` transition matrix
"""
function coarsen(A::AbstractMatrix, from_domain::ContinuousDomain, to_domain::ContinuousDomain)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch(
        "coarsen expects a square matrix, got size $(size(A))"))
    n_fine = from_domain.n_meshpoints
    n_coarse = to_domain.n_meshpoints
    n_fine == size(A, 1) || throw(DimensionMismatch(
        "Matrix size $(size(A, 1)) does not match from_domain meshpoints $n_fine"))
    n_fine % n_coarse == 0 || throw(ArgumentError(
        "n_fine ($n_fine) must be a multiple of n_coarse ($n_coarse)"))
    bins_per_coarse = n_fine ÷ n_coarse

    # fine bin i belongs to coarse bin ceil(i / bins_per_coarse)
    A_coarse = zeros(float(eltype(A)), n_coarse, n_coarse)
    for j_fine in 1:n_fine
        c_j = (j_fine - 1) ÷ bins_per_coarse + 1
        for i_fine in 1:n_fine
            c_i = (i_fine - 1) ÷ bins_per_coarse + 1
            A_coarse[c_i, c_j] += A[i_fine, j_fine] / bins_per_coarse
        end
    end
    return A_coarse
end

"""
    compose_from_uwd(uwd, sub_kernels::Dict, domain)

Evaluate a Catlab undirected wiring diagram (UWD) with concrete sub-kernels: each box is
looked up by name in `sub_kernels`, and the composition sums their contributions on the
mesh of `domain`.

The core package only declares the function. Methods are added where Catlab is available:
by the Catlab extension of IntegralProjectionModels.jl for a `ContinuousDomain`, and by
CategoricalPopulationDynamics.jl for its own domains.

# Arguments
- `uwd`: a Catlab UWD (from `@relation`)
- `sub_kernels`: a `Dict{Symbol, Function}` mapping box names to kernel functions
- `domain`: the domain to discretise on

# Returns
- the composed, discretised transition matrix
"""
function compose_from_uwd end
