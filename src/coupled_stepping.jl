"""
Coupled population systems: the stepping engine shared by the discrete-time model packages.

The data layer (`coupled_systems.jl`) holds a system, its events, rules and observables.
This file holds what steps it one time step at a time: the ordered substeps run before the
components are stepped, the normalised effects of the rules (stage fluxes and transfers)
and their application with a mass-balance check, and the loop over time steps.

How a component is stepped is the one thing left to the solving package. It passes a
function `step_matrix(component, system, day, p)` that returns the operator of a component
for the step (a projection matrix for MatrixProjectionModels.jl, the discretised kernel for
IntegralProjectionModels.jl). `advance_coupled!` advances a system one step and
`solve_coupled` iterates it over a time span.

Nothing in this file is exported from the core package. Each model package imports the
names it uses and exports them, as for the data layer.
"""

# ---------------------------------------------------------------------------
# Ordered daily substeps
# ---------------------------------------------------------------------------

"""
    AbstractDailySubstep

Ordered callbacks executed after scheduled events and before matrix
materialization. Substeps mutate auxiliary state and/or component populations to
support hybrid daily workflows.
"""
abstract type AbstractDailySubstep end

function apply_substep! end

"""
    StateUpdateSubstep(name, state_key, update_fn)

Set `system[state_key] = update_fn(system, day, p)` and record the new value.
"""
struct StateUpdateSubstep{F} <: AbstractDailySubstep
    name::Symbol
    state_key::Symbol
    update_fn::F
end

function apply_substep!(step::StateUpdateSubstep, sys::PopulationSystem, day::Int, p)
    value = deepcopy(step.update_fn(sys, day, p))
    set_state!(sys, step.state_key, value)
    return NamedTuple{(step.state_key,)}((deepcopy(value),))
end

"""
    CustomSubstep(name, apply_fn)

Arbitrary ordered pre-step callback. `apply_fn(system, day, p)` may mutate the
system and should return either a `NamedTuple` of metrics or `nothing`.
"""
struct CustomSubstep{F} <: AbstractDailySubstep
    name::Symbol
    apply_fn::F
end

function apply_substep!(step::CustomSubstep, sys::PopulationSystem, day::Int, p)
    result = step.apply_fn(sys, day, p)
    result === nothing && return NamedTuple()
    result isa NamedTuple || throw(ArgumentError(
        "custom substeps must return a NamedTuple or nothing, got $(typeof(result))"))
    return deepcopy(result)
end

_substep_name(step::AbstractDailySubstep) = getfield(step, :name)

# ---------------------------------------------------------------------------
# Fluxes, rules, events, and observables
# ---------------------------------------------------------------------------

"""
    StageFlux(component, stage, amount)

Signed stage-targeted population change applied after rule evaluation.
Positive values add individuals; negative values remove them.
"""
struct StageFlux{T<:Real}
    component::Symbol
    stage::Int
    amount::T

    function StageFlux{T}(component::Symbol, stage::Int, amount::T) where {T<:Real}
        stage >= 1 || throw(ArgumentError("stage must be >= 1"))
        isfinite(amount) || throw(ArgumentError("amount must be finite"))
        new{T}(component, stage, amount)
    end
end

StageFlux(component::Symbol, stage::Int, amount::Real) =
    StageFlux{typeof(float(amount))}(component, stage, float(amount))

"""
    StageTransfer(source, source_stage, target, target_stage, amount)

Stage-targeted transfer applied after rule evaluation with mass-balance checks.
"""
struct StageTransfer{T<:Real}
    source::Symbol
    source_stage::Int
    target::Symbol
    target_stage::Int
    amount::T

    function StageTransfer{T}(source::Symbol, source_stage::Int,
            target::Symbol, target_stage::Int, amount::T) where {T<:Real}
        source_stage >= 1 || throw(ArgumentError("source_stage must be >= 1"))
        target_stage >= 1 || throw(ArgumentError("target_stage must be >= 1"))
        isfinite(amount) || throw(ArgumentError("amount must be finite"))
        amount >= zero(T) || throw(ArgumentError("transfer amount must be nonnegative"))
        new{T}(source, source_stage, target, target_stage, amount)
    end
end

StageTransfer(source::Symbol, source_stage::Int, target::Symbol, target_stage::Int, amount::Real) =
    StageTransfer{typeof(float(amount))}(source, source_stage, target, target_stage, float(amount))

"""
    RuleEffect(; fluxes = StageFlux[], transfers = StageTransfer[], metrics = NamedTuple())

Normalized output of a transition rule.
"""
struct RuleEffect{T<:Real, M<:NamedTuple}
    fluxes::Vector{StageFlux{T}}
    transfers::Vector{StageTransfer{T}}
    metrics::M
end

function RuleEffect(; fluxes::AbstractVector = StageFlux[],
        transfers::AbstractVector = StageTransfer[],
        metrics = NamedTuple())
    metrics isa NamedTuple || throw(ArgumentError("metrics must be a NamedTuple"))

    amount_types = Type[]
    for flux in fluxes
        flux isa StageFlux || throw(ArgumentError("fluxes must contain StageFlux values"))
        push!(amount_types, typeof(flux.amount))
    end
    for transfer in transfers
        transfer isa StageTransfer || throw(ArgumentError("transfers must contain StageTransfer values"))
        push!(amount_types, typeof(transfer.amount))
    end

    T = isempty(amount_types) ? Float64 : promote_type(amount_types...)
    flux_vec = StageFlux{T}[StageFlux(flux.component, flux.stage, T(flux.amount)) for flux in fluxes]
    transfer_vec = StageTransfer{T}[StageTransfer(transfer.source, transfer.source_stage,
        transfer.target, transfer.target_stage, T(transfer.amount)) for transfer in transfers]
    return RuleEffect{T, typeof(metrics)}(flux_vec, transfer_vec, metrics)
end

function _normalize_rule_effect(result)
    result === nothing && return RuleEffect()
    result isa RuleEffect && return result
    result isa StageFlux && return RuleEffect(fluxes = [result])
    result isa StageTransfer && return RuleEffect(transfers = [result])

    if result isa AbstractVector
        all(x -> x isa StageFlux, result) && return RuleEffect(fluxes = result)
        all(x -> x isa StageTransfer, result) && return RuleEffect(transfers = result)
        throw(ArgumentError("vectors returned by rules must contain only StageFlux or only StageTransfer values"))
    end

    if result isa NamedTuple
        names = propertynames(result)
        if any(name -> name in (:fluxes, :transfers, :metrics), names)
            fluxes = hasproperty(result, :fluxes) ? getproperty(result, :fluxes) : StageFlux[]
            transfers = hasproperty(result, :transfers) ? getproperty(result, :transfers) : StageTransfer[]
            metrics = hasproperty(result, :metrics) ? getproperty(result, :metrics) : NamedTuple()
            return RuleEffect(; fluxes = fluxes, transfers = transfers, metrics = metrics)
        end
        return RuleEffect(metrics = result)
    end

    throw(ArgumentError(
        "rule outputs must be RuleEffect, StageFlux, StageTransfer, vectors of them, or NamedTuples"))
end

"""
    AbstractTransitionRule

The supertype of the rules of a coupled problem. It is the shared
`AbstractInteractionRule` under the name MatrixProjectionModels.jl has always used.
"""
const AbstractTransitionRule = AbstractInteractionRule

"""
    apply_rule(rule, system, day, p) -> RuleEffect

Evaluate a rule on the stepped system and return the changes it asks for, without
applying them. The solver applies the effects of all rules together, with a
mass-balance check. Rule functions are called as `fn(system, day, p)`.
"""
function apply_rule end

function apply_rule(rule::TransferRule, sys::PopulationSystem, day::Int, p)
    comp = sys[rule.source]
    _validate_stage(comp, rule.source_stage)
    _validate_stage(sys[rule.target], rule.target_stage)
    frac = rule.fraction_fn(sys, day, p)
    _validate_fraction("transfer fraction", frac)
    amount = comp.population[rule.source_stage] * frac
    return RuleEffect(
        transfers = [StageTransfer(rule.source, rule.source_stage, rule.target, rule.target_stage, amount)],
        metrics = (transferred = amount,))
end

function apply_rule(rule::ReproductionRule, sys::PopulationSystem, day::Int, p)
    _validate_stage(sys[rule.target], rule.stage)
    offspring = rule.reproduction_fn(sys, day, p)
    _validate_nonnegative("offspring", offspring)
    return RuleEffect(
        fluxes = [StageFlux(rule.target, rule.stage, offspring)],
        metrics = (offspring = offspring,))
end

function apply_rule(rule::MortalityRule, sys::PopulationSystem, day::Int, p)
    comp = sys[rule.target]
    frac = rule.mortality_fn(sys, day, p)
    _validate_fraction("mortality fraction", frac)

    fluxes = StageFlux[]
    if rule.stage == 0
        for stage in eachindex(comp.population)
            push!(fluxes, StageFlux(rule.target, stage, -comp.population[stage] * frac))
        end
    else
        _validate_stage(comp, rule.stage)
        push!(fluxes, StageFlux(rule.target, rule.stage, -comp.population[rule.stage] * frac))
    end

    removed = -sum(flux.amount for flux in fluxes)
    return RuleEffect(fluxes = fluxes, metrics = (removed = removed, mortality = frac))
end

# `apply_fn(system, day, p)` must return one of the accepted `RuleEffect`-compatible outputs.
apply_rule(rule::CustomRule, sys::PopulationSystem, day::Int, p) =
    _normalize_rule_effect(rule.apply_fn(sys, day, p))

"""
    CustomEvent(name, apply_fn)

Custom scheduled event. `apply_fn(system, day, p)` must mutate the system if it
fires and return `true`; otherwise return `false`.
"""
struct CustomEvent{F} <: AbstractScheduledEvent
    name::Symbol
    apply_fn::F
end

function apply_event!(ev::CustomEvent, sys::PopulationSystem, day::Int, p)
    result = ev.apply_fn(sys, day, p)
    result isa Bool || throw(ArgumentError("custom events must return Bool, got $(typeof(result))"))
    return result
end

_rule_name(rule::AbstractTransitionRule) = getfield(rule, :name)
_event_name(event::AbstractScheduledEvent) = getfield(event, :name)

# ---------------------------------------------------------------------------
# Helpers of the problem types
# ---------------------------------------------------------------------------

function _copy_component(comp::PopulationComponent)
    PopulationComponent(comp.model, copy(comp.population);
        stage_names = comp.stage_names,
        species = comp.species,
        type = comp.type,
        patch = comp.patch)
end

function _copy_system(sys::PopulationSystem)
    return PopulationSystem(
        (name => _copy_component(sys[name]) for name in sys.order)...;
        state = deepcopy(sys.state))
end

# A coupled problem steps each component as an operator applied to a stage vector, so every
# component must carry its numbers in one. `problem` names the problem type in the error.
function _require_vector_carriers(sys::PopulationSystem, problem::AbstractString = "CoupledMPMProblem")
    for (name, comp) in pairs(sys)
        comp.population isa AbstractVector{<:Real} || throw(ArgumentError(
            "component :$name carries its population in a $(typeof(comp.population)); " *
            "$problem needs a stage vector"))
    end
    return nothing
end

function _ensure_unique_names(items, label::AbstractString)
    names = Symbol[]
    for item in items
        hasfield(typeof(item), :name) || continue
        name = getfield(item, :name)
        name in names && throw(ArgumentError("duplicate $label name: $name"))
        push!(names, name)
    end
end

# ---------------------------------------------------------------------------
# One time step
# ---------------------------------------------------------------------------

function _snapshot(sys::PopulationSystem{T}) where {T}
    return Dict{Symbol, Vector{T}}(name => copy(sys[name].population) for name in sys.order)
end

function _apply_rule_effects!(sys::PopulationSystem{T}, effects::AbstractVector{<:RuleEffect}) where {T}
    deltas = Dict{Symbol, Vector{T}}(name => zeros(T, length(sys[name].population)) for name in sys.order)

    for effect in effects
        for transfer in effect.transfers
            haskey(sys, transfer.source) || throw(ArgumentError("unknown transfer source :$(transfer.source)"))
            haskey(sys, transfer.target) || throw(ArgumentError("unknown transfer target :$(transfer.target)"))
            _validate_stage(sys[transfer.source], transfer.source_stage)
            _validate_stage(sys[transfer.target], transfer.target_stage)
            amount = T(transfer.amount)
            deltas[transfer.source][transfer.source_stage] -= amount
            deltas[transfer.target][transfer.target_stage] += amount
        end

        for flux in effect.fluxes
            haskey(sys, flux.component) || throw(ArgumentError("unknown flux component :$(flux.component)"))
            _validate_stage(sys[flux.component], flux.stage)
            deltas[flux.component][flux.stage] += T(flux.amount)
        end
    end

    for name in sys.order
        new_population = sys[name].population .+ deltas[name]
        any(value -> value < zero(T), new_population) && throw(ArgumentError(
            "rule effects would make component :$name negative"))
    end

    for name in sys.order
        sys[name].population .+= deltas[name]
    end

    return nothing
end

function _normalize_components!(sys::PopulationSystem)
    for name in sys.order
        pop = sys[name].population
        total = sum(pop)
        total > zero(total) || continue
        pop ./= total
    end
    return nothing
end

# The default stepping of a component: its operator times its stage vector.
_project_population(name::Symbol, A, n::AbstractVector, sys::PopulationSystem, day::Int, p) = Vector(A * n)

"""
    advance_coupled!(step_matrix, system, day, p, events, substeps, rules;
                     normalize = false, event_log = nothing, save_matrices = true,
                     step_population = (name, A, n, system, day, p) -> A * n, saved_matrix = Matrix)

Advance a coupled system one time step, in place, and return `(matrices, substep_metrics,
effects)`. In order:

1. the events that are due fire, and are recorded in `event_log` (a vector of
   `(day, name)`) if one is given;
2. the substeps run, in order, and their metrics are collected;
3. every component is stepped from the state the substeps left: `A = step_matrix(component,
   system, day, p)` is its operator for the step, and its new stage vector is
   `step_population(name, A, n, system, day, p)`, by default `A * n`. All components are
   stepped from the same state and updated together. With `save_matrices`,
   `saved_matrix(A)` is kept for each component;
4. the rules are evaluated on the stepped system and their effects applied together, with
   a mass-balance check;
5. with `normalize`, each component is scaled to sum to one.

`step_matrix` is supplied by the solving package; `step_population` and `saved_matrix` let
it step a component by other means than a product (a demographic draw, say).
"""
function advance_coupled!(step_matrix, sys::PopulationSystem, day::Int, p, events, substeps, rules;
        normalize::Bool = false,
        event_log = nothing,
        save_matrices::Bool = true,
        step_population = _project_population,
        saved_matrix = Matrix)
    for event in events
        fired = apply_event!(event, sys, day, p)
        if fired && event_log !== nothing
            push!(event_log, (day, _event_name(event)))
        end
    end

    substep_metrics = NamedTuple[]
    for substep in substeps
        metrics = apply_substep!(substep, sys, day, p)
        metrics isa NamedTuple || throw(ArgumentError(
            "substeps must return NamedTuple metrics, got $(typeof(metrics))"))
        push!(substep_metrics, deepcopy(metrics))
    end

    current = _snapshot(sys)
    matrices = Dict{Symbol, Any}()
    stepped = Dict{Symbol, Any}()

    for name in sys.order
        comp = sys[name]
        A = step_matrix(comp, sys, day, p)
        save_matrices && (matrices[name] = saved_matrix(A))
        stepped[name] = step_population(name, A, current[name], sys, day, p)
    end

    for name in sys.order
        sys[name].population .= stepped[name]
    end

    effects = RuleEffect[apply_rule(rule, sys, day, p) for rule in rules]
    _apply_rule_effects!(sys, effects)
    normalize && _normalize_components!(sys)

    return matrices, substep_metrics, effects
end

"""
    solve_coupled(step_matrix, system, tspan, p, substeps, rules, events, observables, normalize;
                  save_matrices = true, step_population, saved_matrix)

Iterate `advance_coupled!` on a copy of `system` from `tspan[1]` to `tspan[2]` and
return a `NamedTuple` with the fields of a coupled solution: `t`, `u` (the state of every
component at each time point), `component_names`, `component_matrices` (for each component
the matrix of each step, or an empty vector without `save_matrices`), `component_totals`,
`substep_log`, `observables`, `event_log`, `rule_log` and `retcode`. Observables are
called as `fn(system, t, p)` at every time point.
"""
function solve_coupled(step_matrix, system::PopulationSystem, tspan::Tuple{Int, Int}, p,
        substeps, rules, events, observables, normalize::Bool;
        save_matrices::Bool = true,
        step_population = _project_population,
        saved_matrix = Matrix)
    sys = _copy_system(system)
    t0, tf = tspan
    n_steps = tf - t0
    ts = collect(t0:tf)

    snapshots = Vector{Dict{Symbol, Vector{eltype(first(values(sys.components)).population)}}}(undef, n_steps + 1)
    snapshots[1] = _snapshot(sys)

    component_totals_store = Dict{Symbol, Vector{eltype(first(values(sys.components)).population)}}(
        name => Vector{eltype(first(values(sys.components)).population)}(undef, n_steps + 1)
        for name in sys.order)
    for name in sys.order
        component_totals_store[name][1] = sum(sys[name].population)
    end

    observable_store = Dict{Symbol, Vector{Any}}(
        obs.name => Vector{Any}(undef, n_steps + 1) for obs in observables)
    for obs in observables
        observable_store[obs.name][1] = obs.fn(sys, t0, p)
    end

    component_matrices = Dict{Symbol, Vector{Any}}(
        name => Vector{Any}(undef, save_matrices ? n_steps : 0) for name in sys.order)
    substep_log = Dict{Symbol, Vector{Any}}(
        _substep_name(substep) => Vector{Any}(undef, n_steps) for substep in substeps)
    rule_log = Dict{Symbol, Vector{Any}}(
        _rule_name(rule) => Vector{Any}(undef, n_steps) for rule in rules)
    event_log = Tuple{Int, Symbol}[]

    for step in 1:n_steps
        day = t0 + step - 1
        matrices, substep_metrics, effects = advance_coupled!(
            step_matrix,
            sys,
            day,
            p,
            events,
            substeps,
            rules;
            normalize = normalize,
            event_log = event_log,
            save_matrices = save_matrices,
            step_population = step_population,
            saved_matrix = saved_matrix)

        for name in sys.order
            save_matrices && (component_matrices[name][step] = matrices[name])
            component_totals_store[name][step + 1] = sum(sys[name].population)
        end

        for (substep, metrics) in zip(substeps, substep_metrics)
            substep_log[_substep_name(substep)][step] = metrics
        end

        for (rule, effect) in zip(rules, effects)
            rule_log[_rule_name(rule)][step] = effect.metrics
        end

        snapshots[step + 1] = _snapshot(sys)
        for obs in observables
            observable_store[obs.name][step + 1] = obs.fn(sys, ts[step + 1], p)
        end
    end

    return (t = ts,
            u = snapshots,
            component_names = copy(sys.order),
            component_matrices = component_matrices,
            component_totals = component_totals_store,
            substep_log = substep_log,
            observables = observable_store,
            event_log = event_log,
            rule_log = rule_log,
            retcode = :Success)
end
