using Random
using Statistics

@testset "Individual-level samplers" begin
    rng = Random.Xoshiro(123)
    dom = ContinuousDomain(0.0, 10.0, 200)
    normpdf(x, mu, sd) = exp(-0.5 * ((x - mu) / sd)^2) / (sd * sqrt(2π))

    @testset "sample_survives (Bernoulli, clamped)" begin
        xs = [sample_survives(rng, z -> 0.7, 1.0) for _ in 1:20000]
        @test isapprox(mean(xs), 0.7; atol=0.02)
        @test sample_survives(rng, z -> 2.0, 0.0) == true     # clamp > 1
        @test sample_survives(rng, z -> -1.0, 0.0) == false   # clamp < 0
    end

    @testset "sample_from_density (inverse-CDF on the mesh)" begin
        dens = x -> max(0.0, 1 - abs(x - 7.0) / 3)            # triangular, peak 7
        xs = [sample_from_density(rng, dens, dom) for _ in 1:20000]
        @test isapprox(mean(xs), 7.0; atol=0.1)
        @test all(x -> 0 <= x <= 10, xs)
        @test 0 <= sample_from_density(rng, x -> 0.0, dom) <= 10   # degenerate -> in-domain
    end

    @testset "growth / fecundity generic fallbacks" begin
        growth = (z_new, z) -> normpdf(z_new, z + 1.0, 0.5)
        gs = [sample_growth(rng, growth, 4.0, dom) for _ in 1:20000]
        @test isapprox(mean(gs), 5.0; atol=0.1)

        fec = (z_new, z) -> 2.0 * normpdf(z_new, 2.0, 0.5)    # mean count 2, recruit N(2, 0.5)
        @test isapprox(expected_offspring(fec, 4.0, dom), 2.0; atol=0.02)
        ks = [offspring_count(rng, fec, 4.0, dom) for _ in 1:20000]
        @test isapprox(mean(ks), 2.0; atol=0.05)
        rs = [sample_recruit(rng, fec, 4.0, dom) for _ in 1:20000]
        @test isapprox(mean(rs), 2.0; atol=0.1)
    end
end
