using Random
using Statistics

@testset "Demographic stochasticity" begin
    rng = Random.Xoshiro(20240610)

    @testset "rand_poisson mean/variance" begin
        for λ in (2.0, 50.0)              # small (Knuth) and large (Normal approx)
            xs = [rand_poisson(rng, λ) for _ in 1:20000]
            @test isapprox(mean(xs), λ; rtol=0.05)
            @test isapprox(var(xs), λ; rtol=0.15)
            @test all(>=(0), xs)
        end
        @test rand_poisson(rng, 0.0) == 0
        @test rand_poisson(rng, -1.0) == 0
    end

    @testset "rand_binomial mean/variance" begin
        for n in (20, 500)                 # exact (≤64) and Normal-approx branches
            xs = [rand_binomial(rng, n, 0.3) for _ in 1:20000]
            @test isapprox(mean(xs), n * 0.3; rtol=0.03)
            @test isapprox(var(xs), n * 0.3 * 0.7; rtol=0.12)
            @test all(x -> 0 <= x <= n, xs)
        end
        @test rand_binomial(rng, 10, 0.0) == 0
        @test rand_binomial(rng, 10, 1.0) == 10
        @test rand_binomial(rng, 0, 0.5) == 0
    end

    @testset "demographic_step! conditional mean = operator" begin
        U = [0.0 0.0; 0.6 0.5]          # sub-stochastic survival/growth
        F = [2.0 1.0; 0.0 0.0]          # fecundity
        n = [100, 100]
        reps = 5000
        acc = zeros(2)
        nn = zeros(Int, 2)
        for _ in 1:reps
            demographic_step!(rng, nn, n, U, F)
            @test all(>=(0), nn)
            acc .+= nn
        end
        @test isapprox(acc ./ reps, (U .+ F) * n; rtol=0.05)

        # allocating form; pure mortality column (sum < 1) loses individuals
        out = demographic_step(rng, [10, 0], U, F)
        @test length(out) == 2 && all(>=(0), out)
    end

    @testset "demographic_step! multi-class stick-breaking Multinomial" begin
        # 4 source/destination classes exercises the sequential-Binomial
        # partition across more than one non-trivial destination per column,
        # including a column that sums to exactly 1 (no mortality) and one
        # with a zero-probability destination.
        U = [0.1 0.0 0.0 0.0;
             0.3 0.2 0.0 0.0;
             0.0 0.5 0.4 0.0;
             0.0 0.0 0.6 1.0]
        F = zeros(4, 4)
        n = [1000, 800, 600, 400]
        reps = 3000
        acc = zeros(4)
        nn = zeros(Int, 4)
        for _ in 1:reps
            demographic_step!(rng, nn, n, U, F)
            @test all(>=(0), nn)
            @test sum(nn) <= sum(n)  # no column exceeds its stochastic bound
            acc .+= nn
        end
        @test isapprox(acc ./ reps, U * n; rtol=0.05)
    end

    @testset "demographic_step! large population performance/correctness" begin
        # The stick-breaking rewrite must stay O(k) per source class regardless
        # of population size; this both documents that and checks correctness
        # at a population scale where the old O(n) per-individual loop would
        # have been noticeably slower.
        U = [0.0 0.0; 0.7 0.4]
        F = [3.0 0.5; 0.0 0.0]
        n = [1_000_000, 1_000_000]
        nn = zeros(Int, 2)
        elapsed = @elapsed demographic_step!(rng, nn, n, U, F)
        @test elapsed < 1.0
        @test all(>=(0), nn)
        expected = (U .+ F) * n
        @test isapprox(nn, expected; rtol=0.02)
    end

    @testset "reaction IR construction" begin
        rx = DemographicReaction(1.5, 3, 1 => -1, 2 => +1)
        @test rx.stoichiometry == [-1, 1, 0]
        sys = DemographicReactionSystem(3, [rx])
        @test total_propensity(sys, [5, 0, 0], nothing, 0.0) == 1.5
        n = [5, 0, 0]
        apply_reaction!(n, rx)
        @test n == [4, 1, 0]
    end

    @testset "gillespie: pure death decays as N0 exp(-d t)" begin
        d = 0.5
        sys = DemographicReactionSystem(1,
            [DemographicReaction((n, p, t) -> d * n[1], 1, 1 => -1)])
        tend = 2.0
        finals = [gillespie(rng, sys, [200], (0.0, tend))[2][end][1] for _ in 1:2000]
        @test isapprox(mean(finals), 200 * exp(-d * tend); rtol=0.05)
        @test all(>=(0), finals)
    end

    @testset "gillespie: pure birth (Yule) grows as N0 exp(b t)" begin
        b = 0.3
        sys = DemographicReactionSystem(1,
            [DemographicReaction((n, p, t) -> b * n[1], 1, 1 => +1)])
        tend = 2.0
        finals = [gillespie(rng, sys, [10], (0.0, tend))[2][end][1] for _ in 1:2000]
        @test isapprox(mean(finals), 10 * exp(b * tend); rtol=0.06)
    end

    @testset "generator_reactions" begin
        G = [-0.8 0.4; 0.8 -0.4]                       # conservative
        sys = generator_reactions(G)
        @test sys isa DemographicReactionSystem && sys.n_states == 2
        @test num_reactions(sys) == 2                   # two migrations, zero column sums
        # negative off-diagonal is rejected
        @test_throws ArgumentError generator_reactions([-0.5 -0.2; 0.0 -0.3])
        # constant source adds immigration reactions
        @test num_reactions(generator_reactions(G; source=[1.0, 0.0])) == 3
    end

    @testset "chemical Langevin drift/noise" begin
        G = [-0.5 0.3; 0.5 -0.6]
        src = [1.0, 0.0]
        sys = generator_reactions(G; source=src)
        u = [10.0, 5.0]
        du = zeros(2); cle_drift!(du, sys, u, nothing, 0.0)
        @test isapprox(du, G * u .+ src; atol=1e-10)    # drift = deterministic RHS
        M = zeros(2, num_reactions(sys)); cle_noise!(M, sys, u, nothing, 0.0)
        D = M * M'
        @test isapprox(D, D'; atol=1e-12)               # symmetric demographic diffusion
        @test all(diag(D) .>= 0)
    end
end
