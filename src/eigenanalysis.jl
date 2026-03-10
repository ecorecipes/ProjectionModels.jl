"""
Eigenanalysis routines shared by IPM and MPM packages.
"""

"""
    eigenanalysis_power(A::AbstractMatrix; maxiter=2000, tol=1e-12)

Compute dominant eigenvalue, stable distribution (right eigenvector),
and reproductive value (left eigenvector) via power iteration.

Returns a named tuple `(lambda, stable_dist, repro_value)`.
Reproductive value is normalized so that `dot(v, w) = 1`.
"""
function eigenanalysis_power(A::AbstractMatrix; maxiter=2000, tol=1e-12)
    n = size(A, 1)
    Te = float(eltype(A))

    # Right eigenvector (stable distribution) via power iteration
    w = rand(Te, n) .+ one(Te)
    w ./= norm(w, 1)
    w_new = similar(w)

    λ = one(Te)
    for _ in 1:maxiter
        mul!(w_new, A, w)
        # Normalize by element with largest absolute value (Rayleigh-style)
        idx = argmax(abs.(w_new))
        λ_new = w_new[idx]
        if abs(λ_new) < eps(Te)
            break
        end
        rmul!(w_new, inv(λ_new))
        if maximum(abs, w_new .- w) < tol
            λ = λ_new
            copyto!(w, w_new)
            break
        end
        λ = λ_new
        copyto!(w, w_new)
    end

    # Make eigenvector non-negative (for non-negative matrices, dominant eigvec is non-neg)
    if all(A .>= 0)
        w .= abs.(w)
    end
    ws = sum(w)
    if ws > 0
        w ./= ws
    end

    # Left eigenvector (reproductive value) via power iteration on A'
    At = transpose(A)
    v = rand(Te, n) .+ one(Te)
    v ./= norm(v, 1)
    v_new = similar(v)

    for _ in 1:maxiter
        mul!(v_new, At, v)
        idx = argmax(abs.(v_new))
        λ_v = v_new[idx]
        if abs(λ_v) < eps(Te)
            break
        end
        rmul!(v_new, inv(λ_v))
        if maximum(abs, v_new .- v) < tol
            copyto!(v, v_new)
            break
        end
        copyto!(v, v_new)
    end

    if all(A .>= 0)
        v .= abs.(v)
    end
    vw = dot(v, w)
    if abs(vw) > eps(Te)
        v ./= vw
    end

    return (lambda=real(λ), stable_dist=w, repro_value=v)
end

"""
    eigenanalysis_full(A::AbstractMatrix)

Full eigendecomposition using LinearAlgebra.eigen. Returns a named tuple
`(lambda, stable_dist, repro_value)`. Reproductive value is normalized
so that `dot(v, w) = 1`.
"""
function eigenanalysis_full(A::AbstractMatrix)
    F = eigen(A)
    # Find dominant eigenvalue (largest real part)
    idx = argmax(real.(F.values))
    λ = real(F.values[idx])

    # Right eigenvector (stable distribution) — no in-place mutation
    w = real.(F.vectors[:, idx])
    w = w ./ sum(w)

    # Left eigenvector from inverse
    Fl = eigen(transpose(A))
    idx_l = argmax(real.(Fl.values))
    v = real.(Fl.vectors[:, idx_l])
    vw = dot(v, w)
    if abs(vw) > eps(Float64)
        v = v ./ vw
    end

    return (lambda=λ, stable_dist=w, repro_value=v)
end
