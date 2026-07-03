"""
Matrix properties: is_irreducible, is_primitive, is_ergodic.
"""

"""
    _all_positive_power(M::AbstractMatrix, power::Integer)

Test whether `M^power` has all strictly positive entries. `M` must be
non-negative and `power >= 1`. Uses exponentiation by squaring, so this costs
`O(log(power))` matrix multiplications instead of `O(power)` — important
since [`is_primitive`](@ref) uses a Wielandt bound of `n^2 - 2n + 2`, which
grows quickly for the larger state spaces common in IPM/PSPM discretizations.
"""
function _all_positive_power(M::AbstractMatrix, power::Integer)
    power >= 1 || throw(ArgumentError("power must be positive, got $power"))
    n = size(M, 1)
    result = Matrix{Float64}(I, n, n)
    base = M
    p = power
    while p > 0
        if isodd(p)
            result = result * base
        end
        p >>= 1
        if p > 0
            base = base * base
        end
    end
    return all(result .> 0)
end

"""
    is_irreducible(A::AbstractMatrix)

Test if matrix A is irreducible. A non-negative matrix is irreducible if
its associated directed graph is strongly connected.
Equivalent: (I + |A|)^(n-1) has all positive entries.
"""
function is_irreducible(A::AbstractMatrix)
    n = size(A, 1)
    n == 1 && return true
    B = Matrix{Float64}(I, n, n) .+ abs.(A)
    return _all_positive_power(B, n - 1)
end

"""
    is_primitive(A::AbstractMatrix)

Test if non-negative matrix A is primitive.
A non-negative irreducible matrix is primitive iff it is aperiodic.
Equivalent: A^(n^2 - 2n + 2) has all positive entries (for irreducible A).
"""
function is_primitive(A::AbstractMatrix)
    n = size(A, 1)
    n == 1 && return A[1, 1] > 0
    !is_irreducible(A) && return false
    power = n^2 - 2*n + 2
    M = float.(abs.(A))
    return _all_positive_power(M, power)
end

"""
    is_ergodic(A::AbstractMatrix)

Test if matrix A is ergodic (irreducible and aperiodic/primitive).
An ergodic matrix converges to a unique stable distribution.
"""
function is_ergodic(A::AbstractMatrix)
    return is_primitive(A)
end
