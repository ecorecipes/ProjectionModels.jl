"""
Matrix properties: is_irreducible, is_primitive, is_ergodic.
"""

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
    M = B
    for _ in 2:(n-1)
        M = M * B
    end
    return all(M .> 0)
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
    B = copy(M)
    for _ in 2:power
        B = B * M
    end
    return all(B .> 0)
end

"""
    is_ergodic(A::AbstractMatrix)

Test if matrix A is ergodic (irreducible and aperiodic/primitive).
An ergodic matrix converges to a unique stable distribution.
"""
function is_ergodic(A::AbstractMatrix)
    return is_primitive(A)
end
