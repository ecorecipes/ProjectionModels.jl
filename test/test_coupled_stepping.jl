# The stepping engine of coupled systems, with components stepped by plain matrices.

@testset "Coupled systems (stepping engine)" begin
    SPC = StructuredPopulationCore
    by_matrix(comp, sys, day, p) = comp.model

    @testset "nothing is exported from the core" begin
        for name in (:AbstractDailySubstep, :apply_substep!, :StateUpdateSubstep, :CustomSubstep,
                :StageFlux, :StageTransfer, :RuleEffect, :AbstractTransitionRule, :apply_rule,
                :CustomEvent, :advance_coupled!, :solve_coupled)
            @test isdefined(SPC, name)
            @test !Base.isexported(SPC, name)
        end
        @test SPC.AbstractTransitionRule === SPC.AbstractInteractionRule
    end

    @testset "rule effects" begin
        @test SPC._normalize_rule_effect(nothing) isa SPC.RuleEffect
        flux = SPC.StageFlux(:a, 1, 2)
        @test flux.amount === 2.0
        @test SPC._normalize_rule_effect(flux).fluxes == [flux]
        transfer = SPC.StageTransfer(:a, 1, :b, 2, 0.5)
        @test SPC._normalize_rule_effect([transfer]).transfers == [transfer]
        @test SPC._normalize_rule_effect((x = 1,)).metrics == (x = 1,)
        effect = SPC._normalize_rule_effect((fluxes = [flux], metrics = (y = 2,)))
        @test effect.fluxes == [flux] && effect.metrics == (y = 2,)
        @test_throws ArgumentError SPC._normalize_rule_effect([flux, transfer])
        @test_throws ArgumentError SPC._normalize_rule_effect(1.0)
        @test_throws ArgumentError SPC.StageFlux(:a, 0, 1.0)
        @test_throws ArgumentError SPC.StageTransfer(:a, 1, :b, 1, -1.0)
    end

    @testset "one step, in order" begin
        A = [0.0 2.0; 0.5 0.8]
        B = fill(0.5, 1, 1)
        make() = SPC.PopulationSystem(:host => SPC.PopulationComponent(A, [10.0, 4.0]),
                                      :guest => SPC.PopulationComponent(B, [6.0]); state = (seen = 0.0,))
        sys = make()
        log = Tuple{Int, Symbol}[]
        release = SPC.SingleDayRelease(:guest, 2.0, 3)                       # fires on day 3 only
        recorder = SPC.CustomSubstep(:recorder, (s, day, p) -> (SPC.set_state!(s, :seen, s[:guest].population[1]);
                                                              (guests = s[:guest].population[1],)))
        moving = SPC.TransferRule(:host, 2, :guest, 1, (s, day, p) -> 0.25; name = :moving)
        adding = SPC.CustomRule(:adding, (s, day, p) -> SPC.StageFlux(:host, 1, p))
        matrices, metrics, effects = SPC.advance_coupled!(by_matrix, sys, 3, 1.0, [release], [recorder], [moving, adding];
                                                          event_log = log)
        @test log == [(3, release.name)]
        @test metrics == [(guests = 8.0,)] && SPC.get_state(sys, :seen) == 8.0
        @test matrices[:host] == A && matrices[:host] !== A && matrices[:guest] == B
        host = A * [10.0, 4.0]
        guest = B * [8.0]
        @test sys[:host].population == [host[1] + 1.0, host[2] - 0.25 * host[2]]
        @test sys[:guest].population == [guest[1] + 0.25 * host[2]]
        @test effects[1].metrics == (transferred = 0.25 * host[2],)

        # without saving matrices, with another way of stepping, and normalised
        sys = make()
        halve(name, M, n, s, day, p) = n ./ 2
        matrices, _, _ = SPC.advance_coupled!(by_matrix, sys, 0, nothing, [], [], []; save_matrices = false,
                                              step_population = halve, normalize = true)
        @test isempty(matrices)
        @test sys[:host].population ≈ [10.0, 4.0] ./ 14 && sys[:guest].population == [1.0]

        # rule effects that would leave a component negative are refused
        sys = make()
        draining = SPC.CustomRule(:draining, (s, day, p) -> [SPC.StageFlux(:guest, 1, -100.0)])
        @test_throws ArgumentError SPC.advance_coupled!(by_matrix, sys, 0, nothing, [], [], [draining])
    end

    @testset "a run" begin
        A = [0.0 2.0; 0.5 0.8]
        sys = SPC.PopulationSystem(:host => SPC.PopulationComponent(A, [10.0, 4.0]),
                                   :guest => SPC.PopulationComponent(fill(0.9, 1, 1), [6.0]))
        culling = SPC.MortalityRule(:guest, (s, day, p) -> p; name = :culling)
        counting = SPC.Observable(:total, (s, t, p) -> SPC.total_population(s))
        pulse = SPC.PulseRelease(:guest, 1.0, 2)
        sol = SPC.solve_coupled(by_matrix, sys, (0, 5), 0.1, [], [culling], [pulse], [counting], false)
        @test sol.t == 0:5 && sol.retcode === :Success && sol.component_names == [:host, :guest]
        @test sys[:guest].population == [6.0]                                  # the system given is not changed
        host, guest = [10.0, 4.0], 6.0
        for k in 1:5
            day = k - 1
            iseven(day) && (guest += 1.0)
            host = A * host
            guest = 0.9 * guest
            guest -= guest * 0.1
            @test sol.u[k + 1][:host] ≈ host && sol.u[k + 1][:guest] ≈ [guest]
            @test sol.component_totals[:host][k + 1] ≈ sum(host)
            @test sol.observables[:total][k + 1] ≈ sum(host) + guest
            @test sol.component_matrices[:host][k] == A
        end
        @test sol.event_log == [(0, pulse.name), (2, pulse.name), (4, pulse.name)]
        @test sol.rule_log[:culling][1].mortality == 0.1
        quiet = SPC.solve_coupled(by_matrix, sys, (0, 5), 0.1, [], [culling], [pulse], [counting], false; save_matrices = false)
        @test quiet.u == sol.u && all(isempty, values(quiet.component_matrices))
        @test length(SPC.solve_coupled(by_matrix, sys, (2, 2), 0.1, [], [], [], [], false).u) == 1
    end

    @testset "helpers of the problem types" begin
        sys = SPC.PopulationSystem(:a => SPC.PopulationComponent(fill(1.0, 1, 1), [1.0]); state = (x = [1],))
        copied = SPC._copy_system(sys)
        @test copied[:a].population == [1.0] && copied[:a].population !== sys[:a].population
        @test SPC.get_state(copied, :x) == [1] && SPC.get_state(copied, :x) !== SPC.get_state(sys, :x)
        @test SPC._require_vector_carriers(sys) === nothing
        @test_throws ArgumentError SPC._ensure_unique_names([SPC.CustomRule(:r, identity), SPC.CustomRule(:r, identity)], "rule")
        @test SPC.apply_event!(SPC.CustomEvent(:never, (s, day, p) -> false), sys, 0, nothing) == false
        @test_throws ArgumentError SPC.apply_event!(SPC.CustomEvent(:bad, (s, day, p) -> 1), sys, 0, nothing)
        @test SPC.apply_substep!(SPC.StateUpdateSubstep(:set, :x, (s, day, p) -> [day]), sys, 7, nothing) == (x = [7],)
        @test SPC.get_state(sys, :x) == [7]
    end
end
