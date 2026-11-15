# A state variable and a population carrier defined outside the core, as a model package would.
struct CoupledTestCounter <: StructuredPopulationCore.AbstractStateVariable
    name::Symbol
    value::Base.RefValue{Float64}
end
StructuredPopulationCore.get_state(c::CoupledTestCounter) = c.value[]
StructuredPopulationCore.set_state!(c::CoupledTestCounter, v) = (c.value[] = 2v)

struct CoupledTestCarrier
    n::Base.RefValue{Float64}
end
StructuredPopulationCore.total_population(c::CoupledTestCarrier) = c.n[]
StructuredPopulationCore.inject!(c::CoupledTestCarrier, ::Int, amount::Real) = (c.n[] += amount)
StructuredPopulationCore.remove_fraction!(c::CoupledTestCarrier, fraction::Real) = (c.n[] *= 1 - fraction)
StructuredPopulationCore.as_component(c::CoupledTestCarrier) =
    StructuredPopulationCore.PopulationComponent{Float64, Nothing, CoupledTestCarrier}(
        nothing, c, [:all], :thing, :default, :default)

@testset "Coupled systems (shared data layer)" begin
    SPC = StructuredPopulationCore

    @testset "nothing is exported from the core" begin
        for name in (:PopulationComponent, :PopulationSystem, :Observable,
                :AbstractStateVariable, :AbstractInteractionRule, :AbstractScheduledEvent,
                :PulseRelease, :SingleDayRelease, :SprayEvent,
                :TransferRule, :ReproductionRule, :MortalityRule, :CustomRule,
                :get_state, :set_state!, :has_state, :total_population,
                :component_total, :component_totals, :by_species, :by_type, :by_patch,
                :inject!, :remove_fraction!, :apply_event!,
                :n_stages, :allocate, :supply_demand_index)
            @test isdefined(SPC, name)
            @test !Base.isexported(SPC, name)
        end
    end

    @testset "components" begin
        A = [0.0 2.0; 0.5 0.0]
        comp = SPC.PopulationComponent(A, [1, 2]; species = :fly, type = :wild, patch = :north)
        @test comp isa SPC.PopulationComponent{Float64}
        @test comp.population == [1.0, 2.0]
        @test comp.stage_names == [:stage_1, :stage_2]
        @test (comp.species, comp.type, comp.patch) == (:fly, :wild, :north)
        @test comp.model === A

        named = SPC.PopulationComponent(A, [1.0, 2.0]; stage_names = [:juv, :ad])
        @test named.stage_names == [:juv, :ad]
        @test (named.species, named.type, named.patch) == (:default, :default, :default)

        # The vector is copied
        v = [1.0, 2.0]
        c = SPC.PopulationComponent(A, v)
        v[1] = 99.0
        @test c.population == [1.0, 2.0]

        # A callable model is not checked against the length
        callable = SPC.PopulationComponent((sys, day, p) -> A, [1.0, 2.0])
        @test callable.stage_names == [:stage_1, :stage_2]

        @test_throws ArgumentError SPC.PopulationComponent(A, Float64[])
        @test_throws DimensionMismatch SPC.PopulationComponent(A, [1.0, 2.0, 3.0])
        @test_throws DimensionMismatch SPC.PopulationComponent(A, [1.0, 2.0]; stage_names = [:a])
        @test_throws DimensionMismatch SPC.PopulationComponent(
            (sys, day, p) -> A, [1.0, 2.0]; stage_names = [:a])
    end

    A = [0.0 2.0; 0.5 0.0]
    make_system(; state = NamedTuple()) = SPC.PopulationSystem(
        :a => SPC.PopulationComponent(A, [1.0, 2.0]; species = :fly, type = :wild, patch = :north),
        :b => SPC.PopulationComponent(A, [3.0, 4.0]; species = :fly, type = :sterile, patch = :south),
        :c => SPC.PopulationComponent(A, [5.0, 6.0]; species = :wasp, type = :wild, patch = :north);
        state = state)

    @testset "systems" begin
        sys = make_system()
        @test sys isa SPC.PopulationSystem{Float64}
        @test length(sys) == 3
        @test keys(sys) == [:a, :b, :c]
        @test haskey(sys, :a) && !haskey(sys, :z)
        @test sys[:b].population == [3.0, 4.0]
        @test [name for (name, _) in pairs(sys)] == [:a, :b, :c]
        @test sprint(show, sys) == "PopulationSystem(3 components: a, b, c)"
        @test sprint(show, make_system(state = (x = 1,))) ==
            "PopulationSystem(3 components: a, b, c, 1 state vars)"

        @test SPC.total_population(sys) == 21.0
        @test SPC.total_population([1.0, 2.0]) == 3.0
        @test SPC.component_total(sys, :b) == 7.0
        @test SPC.component_totals(sys) == Dict(:a => 3.0, :b => 7.0, :c => 11.0)
        @test first.(SPC.by_species(sys, :fly)) == [:a, :b]
        @test first.(SPC.by_type(sys, :wild)) == [:a, :c]
        @test first.(SPC.by_patch(sys, :south)) == [:b]

        # Element types are promoted, and vector carriers are copied
        comp = SPC.PopulationComponent(A, Float32[1, 2])
        mixed = SPC.PopulationSystem(:x => comp, :y => SPC.PopulationComponent(A, [1.0, 2.0]))
        @test mixed isa SPC.PopulationSystem{Float64}
        @test eltype(mixed[:x].population) == Float64
        mixed[:x].population[1] = 50.0
        @test comp.population[1] == 1.0f0

        @test_throws ArgumentError SPC.PopulationSystem()
        @test_throws ArgumentError SPC.PopulationSystem(:x => comp, :x => comp)
        @test_throws ArgumentError SPC.PopulationSystem(:x => [1.0, 2.0])
        @test_throws ArgumentError SPC.as_component(1.0)
        @test_throws ArgumentError SPC.PopulationSystem(:x => comp; state = 3)
    end

    @testset "plain state" begin
        counts = [1, 2]
        sys = make_system(state = (level = 2.0, counts = counts))
        @test SPC.has_state(sys, :level) && !SPC.has_state(sys, :other)
        @test SPC.get_state(sys, :level) == 2.0
        # Plain values are copied in
        counts[1] = 10
        @test SPC.get_state(sys, :counts) == [1, 2]
        @test SPC.set_state!(sys, :level, 3.5) == 3.5
        @test SPC.get_state(sys, :level) == 3.5
        SPC.set_state!(sys, :new, :created)
        @test SPC.get_state(sys, :new) == :created
        @test_throws KeyError SPC.get_state(sys, :missing)

        @test SPC.get_state(make_system(state = Dict("k" => 1)), :k) == 1
        @test SPC.get_state(make_system(state = [:k => 1, :m => 2]), :m) == 2
    end

    @testset "state variables are held by reference" begin
        counter = CoupledTestCounter(:count, Ref(1.0))
        other = CoupledTestCounter(:other, Ref(5.0))
        sys = make_system(state = [counter, other])
        @test sys.state[:count] === counter
        @test collect(keys(sys.state)) ==
            collect(keys(Dict{Symbol, Any}(sv.name => sv for sv in [counter, other])))
        @test SPC.get_state(sys, :count) == 1.0
        SPC.set_state!(sys, :count, 4.0)       # goes through the variable's own method
        @test counter.value[] == 8.0
        @test sys.state[:count] === counter
        counter.value[] = 3.0
        @test SPC.get_state(sys, :count) == 3.0

        # Assigning a state variable replaces the entry
        replacement = CoupledTestCounter(:count, Ref(0.0))
        SPC.set_state!(sys, :count, replacement)
        @test sys.state[:count] === replacement

        @test isempty(make_system(state = SPC.AbstractStateVariable[]).state)
    end

    @testset "inject! and remove_fraction!" begin
        sys = make_system()
        SPC.inject!(sys, :a, 2, 3.0)
        @test sys[:a].population == [1.0, 5.0]
        SPC.inject!(sys, :a, 1.0)
        @test sys[:a].population == [2.0, 5.0]
        @test_throws ArgumentError SPC.inject!(sys, :a, 1, -1.0)
        @test_throws BoundsError SPC.inject!(sys, :a, 3, 1.0)
        @test_throws KeyError SPC.inject!(sys, :z, 1, 1.0)

        SPC.remove_fraction!(sys, :b, 0.5)
        @test sys[:b].population == [1.5, 2.0]
        SPC.remove_fraction!(sys, :b, 2, 0.25)
        @test sys[:b].population == [1.5, 1.5]
        @test_throws ArgumentError SPC.remove_fraction!(sys, :b, 1.5)
        @test_throws ArgumentError SPC.remove_fraction!(sys, :b, NaN)
        @test_throws ArgumentError SPC.remove_fraction!(sys, :b, 1, -0.1)
        @test_throws BoundsError SPC.remove_fraction!(sys, :b, 3, 0.1)

        v = [2.0, 4.0]
        SPC.inject!(v, 1, 1.0)
        SPC.remove_fraction!(v, 0.5)
        @test v == [1.5, 2.0]
    end

    @testset "events" begin
        pulse = SPC.PulseRelease(:a, 2.0, 3)
        @test pulse isa SPC.AbstractScheduledEvent
        @test pulse.start_day == 0                      # shared default
        @test pulse.name == :pulse_a
        @test (pulse.stage_idx, pulse.amount, pulse.interval) == (1, 2.0, 3)
        @test pulse.end_day == typemax(Int)

        positional = SPC.PulseRelease(:a, 2, 5, 7; start_day = 1, end_day = 20, name = :weekly)
        @test (positional.stage_idx, positional.amount, positional.interval) == (2, 5.0, 7)
        @test (positional.start_day, positional.end_day, positional.name) == (1, 20, :weekly)
        @test SPC.PulseRelease(:a, 2, 5, 7).start_day == 0

        @test_throws ArgumentError SPC.PulseRelease(:a, 1.0, 0)
        @test_throws ArgumentError SPC.PulseRelease(:a, -1.0, 2)
        @test_throws ArgumentError SPC.PulseRelease(:a, 1.0, 2; stage_idx = 0)
        @test_throws ArgumentError SPC.PulseRelease(:a, 0, 1.0, 2)

        sys = make_system()
        fired = [SPC.apply_event!(pulse, sys, day, nothing) for day in 0:6]
        @test fired == [true, false, false, true, false, false, true]
        @test sys[:a].population == [7.0, 2.0]
        @test SPC.apply_event!(SPC.PulseRelease(:a, 1.0, 2; start_day = 4, end_day = 6), sys, 2, nothing) == false
        @test SPC.apply_event!(SPC.PulseRelease(:a, 1.0, 2; start_day = 4, end_day = 6), sys, 8, nothing) == false

        single = SPC.SingleDayRelease(:b, 10.0, 5; stage_idx = 2)
        @test single.name == :release_b
        @test SPC.SingleDayRelease(:b, 2, 10, 5) == single
        @test !SPC.apply_event!(single, sys, 4, nothing)
        @test SPC.apply_event!(single, sys, 5, nothing)
        @test sys[:b].population == [3.0, 14.0]
        @test_throws ArgumentError SPC.SingleDayRelease(:b, -1.0, 5)
        @test_throws ArgumentError SPC.SingleDayRelease(:b, 0, 1.0, 5)

        spray = SPC.SprayEvent([:a, :c], [0.5, 1.0], [3, 9])
        @test spray.name == :spray
        @test spray.kill_fractions isa Vector{Float64}
        @test !SPC.apply_event!(spray, sys, 4, nothing)
        @test SPC.apply_event!(spray, sys, 9, nothing)
        @test sys[:a].population == [3.5, 1.0]
        @test sys[:c].population == [0.0, 0.0]
        @test_throws DimensionMismatch SPC.SprayEvent([:a], [0.5, 0.5], [1])
        @test_throws ArgumentError SPC.SprayEvent([:a], [1.5], [1])

        # An unknown target is an error when the event fires, not before
        unknown = SPC.SprayEvent([:nobody], [0.5], [2])
        @test !SPC.apply_event!(unknown, sys, 1, nothing)
        @test_throws ArgumentError SPC.apply_event!(unknown, sys, 2, nothing)

        # The form with an environment argument falls back to the form without
        before = copy(sys[:b].population)
        @test SPC.apply_event!(SPC.SingleDayRelease(:b, 1.0, 7), sys, (T = 20.0,), 7, nothing)
        @test sys[:b].population == before .+ [1.0, 0.0]
    end

    @testset "rules and observables" begin
        f = (args...) -> 0.5

        tr = SPC.TransferRule(:a, :b, f; source_stage = 2, target_stage = 1)
        @test tr isa SPC.AbstractInteractionRule
        @test tr.name == :a_to_b
        @test (tr.source, tr.source_stage, tr.target, tr.target_stage) == (:a, 2, :b, 1)
        @test tr.fraction_fn === f
        positional = SPC.TransferRule(:a, 2, :b, 1, f)
        @test positional == tr
        @test SPC.TransferRule(:a, 2, :b, 1, f; name = :move).name == :move

        rr = SPC.ReproductionRule(:a, f; stage = 2)
        @test (rr.name, rr.target, rr.stage) == (:reproduce_a, :a, 2)
        @test rr.reproduction_fn === f
        @test SPC.ReproductionRule(:a, 2, f) == rr
        @test SPC.ReproductionRule(:a, f).stage == 1
        @test_throws ArgumentError SPC.ReproductionRule(:a, f; stage = 0)
        @test_throws ArgumentError SPC.ReproductionRule(:a, 0, f)

        mr = SPC.MortalityRule(:a, f)
        @test (mr.name, mr.target, mr.stage) == (:mortality_a, :a, 0)
        @test mr.mortality_fn === f
        @test SPC.MortalityRule(:a, f; stage = 2, name = :cull).stage == 2
        @test_throws ArgumentError SPC.MortalityRule(:a, f; stage = -1)

        cr = SPC.CustomRule(:custom, f)
        @test cr isa SPC.AbstractInteractionRule
        @test (cr.name, cr.apply_fn) == (:custom, f)

        obs = SPC.Observable(:total, f)
        @test (obs.name, obs.fn) == (:total, f)
    end

    @testset "functions the model packages extend" begin
        for fn in (SPC.n_stages, SPC.allocate, SPC.supply_demand_index)
            @test fn isa Function
        end
        A2 = [0.0 2.0; 0.5 0.0]
        @test SPC.component_stage_names(A2, 2) == [:stage_1, :stage_2]
        @test SPC.component_stage_names(nothing, 3) == [:stage_1, :stage_2, :stage_3]
        @test_throws DimensionMismatch SPC.component_stage_names(A2, 3)
        @test SPC.validate_component_model(nothing, 3) === nothing
        @test_throws DimensionMismatch SPC.validate_component_model(A2, 3)

        # A carrier that is not a vector is kept by reference and cannot change element type
        carrier = CoupledTestCarrier(Ref(10.0))
        sys = SPC.PopulationSystem(:thing => carrier)
        @test sys[:thing].population === carrier
        @test sys[:thing].model === nothing
        SPC.inject!(sys, :thing, 5.0)
        SPC.remove_fraction!(sys, :thing, 0.2)
        @test SPC.total_population(sys) == 12.0
        @test carrier.n[] == 12.0
        @test SPC.component_totals(sys) == Dict(:thing => 12.0)
        @test SPC.convert_component(sys[:thing], Float64) === sys[:thing]
        @test_throws ArgumentError SPC.convert_component(sys[:thing], Float32)
    end
end
