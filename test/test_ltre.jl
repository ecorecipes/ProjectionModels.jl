using Test
using StructuredPopulationCore
using Random

@testset "LTRE Analysis" begin
    # Test matrices (from exactLTRE R package examples)
    A1 = [0.0 0.0 5.0; 0.8 0.0 0.0; 0.0 0.7 0.2]
    A2 = [0.0 0.0 4.0; 0.9 0.0 0.0; 0.0 0.5 0.3]
    A3 = [0.0 0.0 6.0; 0.4 0.0 0.0; 0.0 0.6 0.25]

    @testset "Classical fixed-design LTRE" begin
        result = ltre(A2, A1)
        @test result isa LTREResult
        @test size(result.contributions) == (3, 3)
        @test size(result.difference) == (3, 3)
        # Sum of contributions should approximate Δλ
        @test abs(sum(result.contributions) - result.delta_lambda) < 0.05
        @test result.delta_lambda ≈ result.lambda_treatment - result.lambda_reference

        # Test midpoint vs reference method
        result_ref = ltre(A2, A1; method=:reference)
        @test abs(sum(result_ref.contributions) - result_ref.delta_lambda) < 0.1

        # Test multiple treatments
        results = ltre([A2, A3], A1)
        @test length(results) == 2

        # Test vector of matrices with mean reference
        results2 = ltre([A1, A2, A3])
        @test length(results2) == 3
    end

    @testset "Classical random-design LTRE" begin
        result = ltre_random([A1, A2, A3])
        @test result isa RandomLTREResult
        @test size(result.contributions) == (9, 9)
        @test result.var_lambda > 0
        @test result.lambda_mean > 0
    end

    @testset "Exact fixed-design LTRE" begin
        result = exact_ltre(A2, A1)
        @test result isa ExactLTREResult
        @test result.method == :fixed
        @test !result.directional
        # Exact LTRE should perfectly recover Δλ
        actual_diff = lambda(A2) - lambda(A1)
        @test abs(sum(result.effects) - actual_diff) < 1e-10

        # Directional version
        result_dir = exact_ltre(A2, A1; directional=true)
        @test result_dir.directional
        @test abs(sum(result_dir.effects) - actual_diff) < 1e-10

        # Limited interaction order
        result_max2 = exact_ltre(A2, A1; maxint=2)
        @test length(result_max2.effects) <= length(result.effects)
    end

    @testset "Exact random-design LTRE" begin
        result = exact_ltre([A1, A2, A3])
        @test result isa ExactLTREResult
        @test result.method == :random
        @test length(result.effects) > 0
    end

    @testset "SNA-LTRE" begin
        A4 = [0.0 0.0 5.5; 0.7 0.0 0.0; 0.0 0.65 0.22]
        result = sna_ltre([A1, A2], [A3, A4])
        @test result isa SNALTREResult
        @test size(result.cont_mean) == (3, 3)
        @test size(result.cont_elasticity) == (3, 3)
        @test size(result.cont_cv) == (3, 3)
        @test size(result.cont_correlation) == (3, 3)
        @test isfinite(result.r_treatment)
        @test isfinite(result.r_reference)
    end

    @testset "Stochastic LTRE" begin
        # Small times/burn_in to keep the O(n^2) sensitivity simulation fast;
        # this is checking plumbing/reproducibility, not statistical precision.
        result = stochastic_ltre([A1, A2], [A3];
            times=400, burn_in=50, rng=Random.Xoshiro(1))
        @test result isa StochasticLTREResult
        @test size(result.cont_mean) == (3, 3)
        @test size(result.cont_sd) == (3, 3)
        @test size(result.mean_diff) == (3, 3)
        @test size(result.sd_diff) == (3, 3)
        @test isfinite(result.lambda_s_treatment)
        @test isfinite(result.lambda_s_reference)

        # Explicit rng makes the simulation reproducible.
        result_a = stochastic_ltre([A1, A2], [A3];
            times=400, burn_in=50, rng=Random.Xoshiro(42))
        result_b = stochastic_ltre([A1, A2], [A3];
            times=400, burn_in=50, rng=Random.Xoshiro(42))
        @test result_a.lambda_s_treatment == result_b.lambda_s_treatment
        @test result_a.lambda_s_reference == result_b.lambda_s_reference
        @test result_a.cont_mean == result_b.cont_mean
    end

    @testset "G-matrix" begin
        G2 = gmatrix(2)
        @test size(G2) == (4, 4)
        @test G2[1, 1] == 1.0
        @test G2[4, 4] == 1.0

        G3 = gmatrix(3)
        @test size(G3) == (8, 8)
    end
end
