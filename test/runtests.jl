using Test
using StructuredPopulationCore
using LinearAlgebra

@testset "StructuredPopulationCore" begin

    @testset "Types" begin
        @test DensityIndependent() isa AbstractDensityDependence
        @test DensityDependent() isa AbstractDensityDependence
        @test Deterministic() isa AbstractStochasticity
        @test StochasticKernelResampled() isa AbstractStochasticity
        @test StochasticParameterResampled() isa AbstractStochasticity
        @test DirectIteration() isa DirectIteration
        @test EigenAnalysis() isa EigenAnalysis
        @test DelayGeneratorTerm(1.0, zeros(2, 2)) isa DelayGeneratorTerm
        @test_throws ArgumentError DelayGeneratorTerm(0.0, zeros(1, 1))
    end

    # Test matrix: Leslie-style 3x3
    A = [0.0 3.0 1.0;
         0.5 0.0 0.0;
         0.0 0.3 0.0]

    @testset "eigenanalysis_power" begin
        ea = eigenanalysis_power(A)
        @test ea.lambda > 0
        @test length(ea.stable_dist) == 3
        @test sum(ea.stable_dist) ≈ 1.0 atol=1e-6
        @test all(ea.stable_dist .>= 0)
        @test dot(ea.repro_value, ea.stable_dist) ≈ 1.0 atol=1e-6
    end

    @testset "eigenanalysis_full" begin
        ea = eigenanalysis_full(A)
        @test ea.lambda > 0
        @test sum(ea.stable_dist) ≈ 1.0 atol=1e-6
        @test dot(ea.repro_value, ea.stable_dist) ≈ 1.0 atol=1e-6
    end

    @testset "eigenanalysis agreement" begin
        ea_p = eigenanalysis_power(A)
        ea_f = eigenanalysis_full(A)
        @test ea_p.lambda ≈ ea_f.lambda atol=1e-8
    end

    @testset "lambda (matrix)" begin
        λ = lambda(A)
        @test λ ≈ eigenanalysis_full(A).lambda atol=1e-10
        # Known: for [0 3; 0.5 0], λ = sqrt(1.5) ≈ 1.2247
        A2 = [0.0 3.0; 0.5 0.0]
        @test lambda(A2) ≈ sqrt(1.5) atol=1e-10
    end

    @testset "stable_distribution (matrix)" begin
        w = stable_distribution(A)
        @test length(w) == 3
        @test sum(w) ≈ 1.0 atol=1e-10
        @test all(w .>= 0)
    end

    @testset "reproductive_value (matrix)" begin
        v = reproductive_value(A)
        w = stable_distribution(A)
        @test length(v) == 3
        @test dot(v, w) ≈ 1.0 atol=1e-6
    end

    @testset "sensitivity (matrix)" begin
        S = sensitivity(A)
        @test size(S) == (3, 3)
        @test all(S .>= -1e-10)
    end

    @testset "elasticity (matrix)" begin
        E = elasticity(A)
        @test size(E) == (3, 3)
        @test sum(E) ≈ 1.0 atol=0.01
    end

    @testset "damping_ratio (matrix)" begin
        ρ = damping_ratio(A)
        @test ρ > 1.0  # dominant should exceed subdominant
        @test isfinite(ρ)
    end

    @testset "is_irreducible" begin
        @test is_irreducible(A) == true
        # Reducible: block diagonal
        B = [1.0 0.0; 0.0 1.0]
        @test is_irreducible(B) == false
    end

    @testset "is_primitive" begin
        @test is_primitive(A) == true
        # Periodic (imprimitive): permutation matrix
        P = [0.0 1.0; 1.0 0.0]
        @test is_primitive(P) == false
    end

    @testset "is_ergodic" begin
        @test is_ergodic(A) == true
    end

    @testset "area_under_curve" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 0.0]
        @test area_under_curve(x, y) ≈ 1.0
        # Single point
        @test area_under_curve([1.0], [2.0]) == 0.0
    end

    include("test_shared_continuous.jl")
    include("test_state_blocks.jl")
    include("test_time_lag.jl")

end
