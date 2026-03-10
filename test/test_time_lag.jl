@testset "Time Lag" begin

    @testset "TimeLagStructure" begin
        tl = TimeLagStructure(1)
        @test tl.max_lag == 1
        tl2 = TimeLagStructure(3)
        @test tl2.max_lag == 3
        @test_throws ArgumentError TimeLagStructure(0)
        @test_throws ArgumentError TimeLagStructure(-1)
    end

    @testset "expand_lag_matrix (L=1)" begin
        # Leslie-style: P = survival, F = fecundity with 1-step delay
        P = [0.0 0.0; 0.5 0.0]
        F = [0.0 3.0; 0.0 0.0]
        K_aug = expand_lag_matrix(P, F)

        @test size(K_aug) == (4, 4)
        # Top-left block = P
        @test K_aug[1:2, 1:2] == P
        # Top-right block = F
        @test K_aug[1:2, 3:4] == F
        # Bottom-left block = I
        @test K_aug[3:4, 1:2] == [1.0 0.0; 0.0 1.0]
        # Bottom-right block = 0
        @test K_aug[3:4, 3:4] == zeros(2, 2)
    end

    @testset "expand_lag_matrix (L=2)" begin
        m = 2
        K0 = [0.5 0.0; 0.3 0.2]
        K1 = [0.0 1.0; 0.0 0.0]
        K2 = [0.0 0.5; 0.0 0.0]
        tl = TimeLagStructure(2)
        K_aug = expand_lag_matrix([K0, K1, K2], tl)

        @test size(K_aug) == (6, 6)
        # Top row blocks
        @test K_aug[1:2, 1:2] == K0
        @test K_aug[1:2, 3:4] == K1
        @test K_aug[1:2, 5:6] == K2
        # First sub-diagonal identity
        @test K_aug[3:4, 1:2] == I(2)
        @test K_aug[3:4, 3:4] == zeros(2, 2)
        @test K_aug[3:4, 5:6] == zeros(2, 2)
        # Second sub-diagonal identity
        @test K_aug[5:6, 1:2] == zeros(2, 2)
        @test K_aug[5:6, 3:4] == I(2)
        @test K_aug[5:6, 5:6] == zeros(2, 2)
    end

    @testset "expand_lag_matrix validation" begin
        P = [0.5 0.0; 0.3 0.2]
        F = [0.0 1.0; 0.0 0.0]
        tl = TimeLagStructure(2)
        # Wrong number of kernels
        @test_throws ArgumentError expand_lag_matrix([P, F], tl)
        # Mismatched dimensions
        F_bad = [0.0 1.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
        @test_throws DimensionMismatch expand_lag_matrix([P, F_bad], TimeLagStructure(1))
    end

    @testset "extract_lag_components" begin
        P = [0.5 0.0; 0.3 0.2]
        F = [0.0 1.0; 0.0 0.0]
        K_aug = expand_lag_matrix(P, F)
        tl = TimeLagStructure(1)
        result = extract_lag_components(K_aug, 2, tl)

        @test length(result.kernels) == 2
        @test result.kernels[1] ≈ P
        @test result.kernels[2] ≈ F
        @test result.m == 2
    end

    @testset "augment_population" begin
        n0 = [10.0, 5.0, 2.0]
        tl = TimeLagStructure(1)
        n_aug = augment_population(n0, tl)
        @test length(n_aug) == 6
        @test n_aug[1:3] == n0
        @test n_aug[4:6] == n0

        tl2 = TimeLagStructure(2)
        n_aug2 = augment_population(n0, tl2)
        @test length(n_aug2) == 9
    end

    @testset "extract_population" begin
        n_aug = [10.0, 5.0, 2.0, 8.0, 4.0, 1.0]
        n = extract_population(n_aug, 3)
        @test n == [10.0, 5.0, 2.0]
    end

    @testset "eigenvalue of augmented matrix" begin
        # Lagged model: n(t+1) = P·n(t) + F·n(t-1) differs from standard A = P + F
        P = [0.0 0.0; 0.5 0.0]
        F = [0.0 3.0; 0.0 0.0]
        K_aug = expand_lag_matrix(P, F)
        λ_aug = lambda(K_aug)
        @test λ_aug > 0
        @test isfinite(λ_aug)

        # Cross-check: lambda should be max real eigenvalue from eigvals
        vals = eigvals(K_aug)
        λ_expected = maximum(real.(vals))
        @test λ_aug ≈ λ_expected atol=1e-10

        # Lagged model λ should differ from non-lagged P+F
        A_standard = P + F
        @test !isapprox(lambda(K_aug), lambda(A_standard); atol=1e-4)
    end

    @testset "L=0 degeneration" begin
        # With no lag, expand_lag_matrix([A], TimeLagStructure) doesn't apply
        # (TimeLagStructure requires max_lag > 0), but we can verify that
        # a lag-1 model with F=0 equals the standard model
        P = [0.0 3.0; 0.5 0.0]
        F = zeros(2, 2)
        K_aug = expand_lag_matrix(P, F)
        # λ of augmented should equal λ of P
        @test lambda(K_aug) ≈ lambda(P) atol=1e-10
    end

    @testset "net_repro_rate_lagged" begin
        P = [0.0 0.0; 0.5 0.0]
        F = [0.0 3.0; 0.0 0.0]
        R0 = net_repro_rate_lagged(P, F)
        @test R0 > 0
        @test isfinite(R0)

        # For diagonal P and diagonal F, R0 should be computable
        P_diag = diagm([0.5, 0.3])
        F_diag = diagm([2.0, 1.5])
        R0_diag = net_repro_rate_lagged(P_diag, F_diag)
        @test R0_diag > 0
        @test isfinite(R0_diag)

        # Multi-lag version should give same result for L=1
        tl = TimeLagStructure(1)
        R0_multi = net_repro_rate_lagged([P, F], tl)
        @test R0_multi ≈ R0 atol=1e-10
    end

    @testset "generation_time_lagged" begin
        P = [0.0 0.0; 0.5 0.0]
        F = [0.0 3.0; 0.0 0.0]
        T = generation_time_lagged(P, F)
        @test T > 0
        @test isfinite(T)
    end

    @testset "multi-lag (L=3)" begin
        m = 3
        K0 = diagm([0.5, 0.3, 0.2])
        K1 = [0.0 0.0 1.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
        K2 = [0.0 0.0 0.5; 0.0 0.0 0.0; 0.0 0.0 0.0]
        K3 = [0.0 0.0 0.2; 0.0 0.0 0.0; 0.0 0.0 0.0]
        tl = TimeLagStructure(3)
        K_aug = expand_lag_matrix([K0, K1, K2, K3], tl)

        @test size(K_aug) == (12, 12)
        λ = lambda(K_aug)
        @test λ > 0
        @test isfinite(λ)

        # Verify extract roundtrip
        result = extract_lag_components(K_aug, m, tl)
        @test result.kernels[1] ≈ K0
        @test result.kernels[2] ≈ K1
        @test result.kernels[3] ≈ K2
        @test result.kernels[4] ≈ K3
    end

end
