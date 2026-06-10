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
end
