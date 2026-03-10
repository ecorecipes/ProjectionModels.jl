"""
Shared utility functions.
"""

"""
    area_under_curve(x, y)

Trapezoidal numerical integration of y over x.
"""
function area_under_curve(x::AbstractVector, y::AbstractVector)
    length(x) == length(y) || throw(DimensionMismatch("x and y must have same length"))
    n = length(x)
    n < 2 && return zero(eltype(y))
    s = zero(promote_type(eltype(x), eltype(y)))
    for i in 1:(n-1)
        s += (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2
    end
    return s
end
