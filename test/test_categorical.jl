# Kernels to matrices and back, stratification and coarsening on the core types, and the verbs
# that the model packages implement.

@testset "categorical operations" begin
    domain = ContinuousDomain(0.0, 10.0, 40)
    z, h = meshpoints(domain), step_size(domain)
    kernel(y, x) = exp(-0.5 * ((y - (1.0 + 0.8x)) / 0.7)^2) / (0.7 * sqrt(2pi)) / (1 + exp(1.0 - 0.5x))

    @testset "left and right Kan extension" begin
        A = left_kan_extension(kernel, domain)
        @test size(A) == (40, 40)
        @test A == [h * kernel(y, x) for y in z, x in z]
        # quadrature options belong to CategoricalPopulationDynamics.jl
        @test_throws ArgumentError left_kan_extension(kernel, domain; rule = :simpson)
        K = right_kan_extension(A, domain)
        @test K(z[7], z[3]) == A[7, 3] / h                    # constant within a pair of bins
        @test K(z[7] + 0.4h, z[3] - 0.4h) == A[7, 3] / h
        @test K(-5.0, 100.0) == A[1, 40] / h                  # clamped to the domain
        @test left_kan_extension(K, domain) ≈ A               # refining, then discretising, gives the matrix back
        @test right_kan_extension(view(A, :, :), domain)(z[2], z[5]) == A[2, 5] / h
    end

    @testset "stratify" begin
        A = [0.0 2.0; 0.5 0.8]
        D = [0.9 0.2; 0.1 0.8]
        S = stratify(A, D)
        @test S == kron(D, A)
        @test S[3:4, 1:2] == 0.1 .* A                         # from stratum 1 to stratum 2
        @test stratify(A, Matrix(1.0I, 3, 3)) == kron(Matrix(1.0I, 3, 3), A)
        @test eltype(stratify([0 1; 1 0], [1 0; 0 1])) == Int  # element types are promoted, not forced
        @test stratify(view(A, :, :), D') == kron(D', A)       # any AbstractMatrix
        @test_throws DimensionMismatch stratify(ones(2, 3), D)
        @test_throws DimensionMismatch stratify(A, ones(2, 3))
    end

    @testset "coarsen" begin
        fine, coarse = ContinuousDomain(0.0, 10.0, 40), ContinuousDomain(0.0, 10.0, 10)
        A = left_kan_extension(kernel, fine)
        C = coarsen(A, fine, coarse)
        @test size(C) == (10, 10)
        # a coarse bin collects its four fine bins; a column is averaged over the fine bins it holds
        @test C[2, 3] ≈ sum(A[5:8, 9:12]) / 4
        @test vec(sum(C; dims = 1)) ≈ [sum(A[:, (4j - 3):(4j)]) / 4 for j in 1:10]
        @test coarsen(A, fine, fine) == A
        @test eltype(coarsen([1 0; 0 1], ContinuousDomain(0.0, 1.0, 2), ContinuousDomain(0.0, 1.0, 1))) == Float64
        @test_throws ArgumentError coarsen(A, fine, ContinuousDomain(0.0, 10.0, 7))
        @test_throws DimensionMismatch coarsen(A[1:10, 1:10], fine, coarse)
        @test_throws DimensionMismatch coarsen(ones(40, 41), fine, coarse)
    end

    @testset "verbs declared for the model packages" begin
        # declared here, without methods, so that every model package extends the same function
        for f in (compose_from_uwd, demographic_ensemble, to_discrete_problem)
            @test f isa Function && isempty(methods(f))
        end
    end
end
