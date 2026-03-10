"""
Time-lagged projection model support.

Implements state augmentation for projection models where some kernels/matrices
act on population states from previous time steps. Following Kuss et al. (2008),
the augmented state `m(t) = (n(t), n(t-1), ..., n(t-L))^T` evolves via a block
matrix:

```
K_aug = [ K_0  K_1  ...  K_L ]
        [  I    0   ...   0  ]
        [  0    I   ...   0  ]
        [  .    .   ...   .  ]
        [  0    0   ...   0  ]
```

where K_k is the kernel/matrix acting on n(t-k), and the identity sub-diagonal
shifts the population history forward.
"""

"""
    TimeLagStructure

Describes the lag structure of a projection model.

# Fields
- `max_lag::Int`: Maximum lag L (must be > 0). The augmented state has L+1 blocks.
"""
struct TimeLagStructure
    max_lag::Int

    function TimeLagStructure(max_lag::Int)
        max_lag > 0 || throw(ArgumentError("max_lag must be positive, got $max_lag"))
        new(max_lag)
    end
end

Base.show(io::IO, tl::TimeLagStructure) = print(io, "TimeLagStructure(max_lag=$(tl.max_lag))")

"""
    expand_lag_matrix(lag_kernels::AbstractVector{<:AbstractMatrix}, lag_structure::TimeLagStructure)

Build the `(L+1)m × (L+1)m` augmented block matrix from a vector of kernel matrices,
one per lag index (0, 1, ..., L).

# Arguments
- `lag_kernels`: Vector of matrices `[K_0, K_1, ..., K_L]` where `K_k` acts on `n(t-k)`.
  Length must equal `lag_structure.max_lag + 1`.
- `lag_structure`: `TimeLagStructure` specifying the maximum lag.

# Returns
The augmented block matrix with kernel matrices in the top row and identity
matrices on the sub-diagonal.
"""
function expand_lag_matrix(lag_kernels::AbstractVector{<:AbstractMatrix},
        lag_structure::TimeLagStructure)
    L = lag_structure.max_lag
    length(lag_kernels) == L + 1 ||
        throw(ArgumentError("Expected $(L + 1) lag kernels, got $(length(lag_kernels))"))

    m = size(lag_kernels[1], 1)
    for (i, K) in enumerate(lag_kernels)
        size(K) == (m, m) ||
            throw(DimensionMismatch("All lag kernels must be $m × $m; kernel $i is $(size(K))"))
    end

    T = promote_type(map(eltype, lag_kernels)...)
    n_total = (L + 1) * m
    K_aug = zeros(T, n_total, n_total)

    # Top block row: [K_0, K_1, ..., K_L]
    for k in 0:L
        cols = (k * m + 1):((k + 1) * m)
        K_aug[1:m, cols] .= lag_kernels[k + 1]
    end

    # Sub-diagonal identity blocks: I at position (block k+1, block k) for k = 0, ..., L-1
    for k in 0:(L - 1)
        rows = ((k + 1) * m + 1):((k + 2) * m)
        cols = (k * m + 1):((k + 1) * m)
        for i in 1:m
            K_aug[rows[i], cols[i]] = one(T)
        end
    end

    return K_aug
end

"""
    expand_lag_matrix(P::AbstractMatrix, F::AbstractMatrix)

Convenience for the common single-lag case (L=1):
`n(t+1) = P·n(t) + F·n(t-1)`.

Returns the `2m × 2m` augmented matrix `[[P, F], [I, 0]]`.
"""
function expand_lag_matrix(P::AbstractMatrix, F::AbstractMatrix)
    expand_lag_matrix([P, F], TimeLagStructure(1))
end

"""
    extract_lag_components(K_aug::AbstractMatrix, m::Int, lag_structure::TimeLagStructure)

Extract the component matrices from an augmented block matrix.

# Returns
A named tuple `(kernels=Vector{Matrix}, m=Int, lag_structure=TimeLagStructure)`
where `kernels[k+1]` is the matrix acting on `n(t-k)`.
"""
function extract_lag_components(K_aug::AbstractMatrix, m::Int, lag_structure::TimeLagStructure)
    L = lag_structure.max_lag
    n_total = (L + 1) * m
    size(K_aug) == (n_total, n_total) ||
        throw(DimensionMismatch("Expected $(n_total)×$(n_total) augmented matrix"))

    kernels = [Matrix(K_aug[1:m, (k * m + 1):((k + 1) * m)]) for k in 0:L]
    return (kernels=kernels, m=m, lag_structure=lag_structure)
end

"""
    augment_population(n0::AbstractVector, lag_structure::TimeLagStructure)

Replicate a population vector into the augmented state format.
All lag slots are initialized with a copy of `n0`.

Returns a vector of length `(L+1) * length(n0)`.
"""
function augment_population(n0::AbstractVector, lag_structure::TimeLagStructure)
    L = lag_structure.max_lag
    return repeat(n0, L + 1)
end

"""
    extract_population(n_aug::AbstractVector, m::Int)

Extract the current physical population state (first `m` elements)
from an augmented state vector.
"""
function extract_population(n_aug::AbstractVector, m::Int)
    return n_aug[1:m]
end

"""
    net_repro_rate_lagged(P::AbstractMatrix, F::AbstractMatrix)

Net reproductive rate R0 for a time-lagged model where P acts on n(t)
and F acts on n(t-1).

Uses the augmented fundamental matrix approach (Kuss et al. 2008):
- U_aug = [[P, 0], [I, 0]] (survival/transition part of augmented system)
- F_aug = [[0, F], [0, 0]] (fecundity part of augmented system)
- N_aug = (I - U_aug)^{-1}
- R0 = ρ(F_aug · N_aug)

Returns `Inf` if the fundamental matrix is singular.
"""
function net_repro_rate_lagged(P::AbstractMatrix, F::AbstractMatrix)
    m = size(P, 1)
    size(P) == (m, m) || throw(DimensionMismatch("P must be square"))
    size(F) == (m, m) || throw(DimensionMismatch("F must be same size as P"))

    n = 2 * m
    # U_aug = [[P, 0], [I, 0]]
    U_aug = zeros(Float64, n, n)
    U_aug[1:m, 1:m] .= P
    for i in 1:m
        U_aug[m + i, i] = 1.0
    end

    # F_aug = [[0, F], [0, 0]]
    F_aug = zeros(Float64, n, n)
    F_aug[1:m, (m + 1):n] .= F

    ImU = Matrix{Float64}(I, n, n) .- U_aug
    d = det(ImU)
    abs(d) < 1e-15 && return Inf

    N_aug = inv(ImU)
    R0_mat = F_aug * N_aug
    vals = eigvals(R0_mat)
    return maximum(abs.(vals))
end

"""
    net_repro_rate_lagged(lag_kernels::AbstractVector{<:AbstractMatrix},
                          lag_structure::TimeLagStructure)

Generalized R0 for multi-lag models. Decomposes the augmented matrix into
survival (U_aug) and fecundity (F_aug) parts, then computes R0 via
the fundamental matrix.

`lag_kernels[1]` is treated as the survival/transition kernel (U),
and `lag_kernels[2:end]` as fecundity kernels at each lag.
"""
function net_repro_rate_lagged(lag_kernels::AbstractVector{<:AbstractMatrix},
        lag_structure::TimeLagStructure)
    L = lag_structure.max_lag
    length(lag_kernels) == L + 1 ||
        throw(ArgumentError("Expected $(L + 1) lag kernels, got $(length(lag_kernels))"))
    m = size(lag_kernels[1], 1)
    n = (L + 1) * m

    # U_aug: survival/transition component
    # Top-left block = lag_kernels[1] (immediate survival)
    # Sub-diagonal = I (history shifts)
    # All fecundity blocks zero
    U_aug = zeros(Float64, n, n)
    U_aug[1:m, 1:m] .= lag_kernels[1]
    for k in 0:(L - 1)
        rows = ((k + 1) * m + 1):((k + 2) * m)
        cols = (k * m + 1):((k + 1) * m)
        for i in 1:m
            U_aug[rows[i], cols[i]] = 1.0
        end
    end

    # F_aug: fecundity components (lag_kernels[2:end])
    F_aug = zeros(Float64, n, n)
    for k in 1:L
        cols = (k * m + 1):((k + 1) * m)
        F_aug[1:m, cols] .= lag_kernels[k + 1]
    end

    ImU = Matrix{Float64}(I, n, n) .- U_aug
    d = det(ImU)
    abs(d) < 1e-15 && return Inf

    N_aug = inv(ImU)
    R0_mat = F_aug * N_aug
    vals = eigvals(R0_mat)
    return maximum(abs.(vals))
end

"""
    generation_time_lagged(P::AbstractMatrix, F::AbstractMatrix)

Generation time T for a time-lagged model: `T = log(R0) / log(λ)`.

`λ` is computed from the augmented matrix `[[P, F], [I, 0]]`.
"""
function generation_time_lagged(P::AbstractMatrix, F::AbstractMatrix)
    R0 = net_repro_rate_lagged(P, F)
    R0 <= 0 && return Inf
    K_aug = expand_lag_matrix(P, F)
    λ = lambda(K_aug)
    λ <= 0 && return Inf
    return log(R0) / log(λ)
end
