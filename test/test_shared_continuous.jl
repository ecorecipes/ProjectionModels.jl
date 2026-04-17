@testset "Shared continuous-state abstractions" begin

    @test AbstractContinuousStateStructure <: AbstractProjectionStructure
    @test AbstractIPMStructure <: AbstractContinuousStateStructure
    @test SimpleIPM() isa AbstractIPMStructure
    @test GeneralIPM() isa AbstractIPMStructure
    @test SimpleContinuousState() isa AbstractContinuousStateStructure
    @test GeneralContinuousState() isa AbstractContinuousStateStructure

    @test DiscreteTime() isa AbstractTimeSemantics
    @test ContinuousTime() isa AbstractTimeSemantics
    @test FiniteState() isa AbstractStateSemantics
    @test ContinuousState() isa AbstractStateSemantics

    d = ContinuousDomain(0.0, 10.0, 5)
    @test d isa AbstractStateDomain
    @test step_size(d) ≈ 2.0
    @test meshpoints(d) == [1.0, 3.0, 5.0, 7.0, 9.0]
    @test bounds(d) == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    @test n_states(d) == 5

    dd = DiscreteDomain([:juvenile, :adult])
    @test dd isa AbstractStateDomain
    @test n_states(dd) == 2

    @test_throws ArgumentError ContinuousDomain(1.0, 1.0, 10)
    @test_throws ArgumentError ContinuousDomain(0.0, 1.0, 0)
end
