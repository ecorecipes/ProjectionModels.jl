"""
Coupled population systems: the data layer shared by the model packages.

A coupled system is an ordered set of named population components plus named auxiliary
state, with scheduled events, interaction rules and observables acting on it.
MatrixProjectionModels.jl steps the components with projection matrices,
PhysiologicallyBasedDemographicModels.jl with distributed delays; the container, the
events, the rule descriptions and the helpers that read and change a system are the same
and are defined once, here.

Nothing in this file is exported from the core package. Each model package imports the
names it uses and exports them, so that a name means the same thing whichever packages are
loaded together.

A component carries its numbers in a *carrier* (`component.population`). The core
implements the carrier interface for plain vectors (one entry per stage). A package with
another kind of carrier adds methods of `total_population(carrier)`,
`inject!(carrier, stage, amount)`, `remove_fraction!(carrier, fraction)`,
`remove_fraction!(carrier, stage, fraction)` and `as_component(carrier)`.
"""

# ---------------------------------------------------------------------------
# Abstract types
# ---------------------------------------------------------------------------

"""
    AbstractStateVariable

Supertype for named auxiliary state held in a [`PopulationSystem`](@ref) by reference and
read or changed through `get_state` and `set_state!`. A package that defines a state
variable type adds `get_state(variable)` and `set_state!(variable, value)` methods for it.
Plain values (numbers, arrays, dictionaries) can be stored as state without a wrapper.
"""
abstract type AbstractStateVariable end

"""
    AbstractInteractionRule

Supertype for rules evaluated at each time step after the components have been stepped
(transfers between components, reproduction, extra mortality, user-defined logic). The
rule types describe what to do; each model package's solver decides how a rule is applied.
"""
abstract type AbstractInteractionRule end

"""
    AbstractScheduledEvent

Supertype for events applied at the start of a time step on particular days (releases,
sprays). `apply_event!(event, system, day, p)` returns `true` if the event fired.
"""
abstract type AbstractScheduledEvent end

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_validate_fraction(label::AbstractString, value::Real) =
    (isfinite(value) && 0 <= value <= 1) ||
    throw(ArgumentError("$label must be finite and lie in [0, 1], got $value"))

function _validate_nonnegative(label::AbstractString, value::Real)
    isfinite(value) || throw(ArgumentError("$label must be finite, got $value"))
    value >= 0 || throw(ArgumentError("$label must be nonnegative, got $value"))
    return value
end

function _validate_stage(population::AbstractVector, stage::Int)
    1 <= stage <= length(population) || throw(BoundsError(
        "stage $stage out of range for component with $(length(population)) stages"))
end

default_stage_names(n::Int) = [Symbol("stage_$i") for i in 1:n]

# ---------------------------------------------------------------------------
# Hooks and functions that the model packages extend
# ---------------------------------------------------------------------------

"""
    n_stages(x)

Number of stages of a model or of a population carrier. Each package adds the method for
its own types.
"""
function n_stages end

"""
    allocate(pool)

Allocate the supply of a resource pool to its demands. Each package adds the method for
its own pool type.
"""
function allocate end

"""
    supply_demand_index(pool)

Overall supply/demand ratio of a resource pool, in `[0, 1]`. Each package adds the method
for its own pool type.
"""
function supply_demand_index end

"""
    as_component(x)

Wrap `x` as a [`PopulationComponent`](@ref). `PopulationSystem` calls it for values that
are not already components. A package with its own population carrier adds a method; the
fallback throws an `ArgumentError`.
"""
as_component(x) = throw(ArgumentError(
    "expected a PopulationComponent, got $(typeof(x))"))

"""
    component_stage_names(model, n)

Stage names of a vector-carried component with `n` stages when none are given. The method
for matrices checks the size and returns `stage_1`, …; a package adds a method for a model
type that has names of its own.
"""
component_stage_names(model, n::Int) = default_stage_names(n)

function component_stage_names(model::AbstractMatrix, n::Int)
    size(model) == (n, n) || throw(DimensionMismatch(
        "model has size $(size(model)); expected ($n, $n)"))
    return default_stage_names(n)
end

"""
    validate_component_model(model, n)

Check that `model` fits a vector-carried component with `n` stages, throwing a
`DimensionMismatch` if it does not. The fallback accepts anything.
"""
validate_component_model(model, n::Int) = nothing

function validate_component_model(model::AbstractMatrix, n::Int)
    size(model) == (n, n) || throw(DimensionMismatch(
        "model has size $(size(model)); expected ($n, $n)"))
end

# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

"""
    PopulationComponent(model, population; stage_names, species, type, patch)

A named stage-structured population component of a [`PopulationSystem`](@ref).

- `model` is what steps the component: a matrix, a model type of one of the packages, or a
  callable `(system, day, p) -> AbstractMatrix`. It is `nothing` when the carrier steps
  itself.
- `population` is the carrier of the current numbers: a stage vector, or a carrier type
  of one of the packages.
- `species`, `type` and `patch` carry grouping metadata.

This constructor is for a stage vector. The vector is copied and converted to floating
point, and the model is checked against its length.
"""
struct PopulationComponent{T<:Real, M, P}
    model::M
    population::P
    stage_names::Vector{Symbol}
    species::Symbol
    type::Symbol
    patch::Symbol
end

function PopulationComponent(model, population::AbstractVector{<:Real};
        stage_names::AbstractVector{Symbol} = Symbol[],
        species::Symbol = :default,
        type::Symbol = :default,
        patch::Symbol = :default)
    pop = collect(float.(population))
    isempty(pop) && throw(ArgumentError("population must contain at least one stage"))
    inferred_stage_names = isempty(stage_names) ?
        component_stage_names(model, length(pop)) :
        collect(stage_names)
    length(inferred_stage_names) == length(pop) || throw(DimensionMismatch(
        "stage_names length $(length(inferred_stage_names)) does not match population length $(length(pop))"))
    validate_component_model(model, length(pop))
    return PopulationComponent{eltype(pop), typeof(model), typeof(pop)}(
        model,
        pop,
        inferred_stage_names,
        species,
        type,
        patch)
end

_component_eltype(::PopulationComponent{T}) where {T} = T

_validate_stage(comp::PopulationComponent, stage::Int) = _validate_stage(comp.population, stage)

"""
    convert_component(component, T)

Return `component` with element type `T`. A vector-carried component is rebuilt with a
converted copy of its vector. Other carriers are kept as they are, by reference, and must
already have element type `T`.
"""
function convert_component(comp::PopulationComponent{S}, ::Type{T}) where {S<:Real, T<:Real}
    S === T || throw(ArgumentError(
        "cannot convert a component carried by $(typeof(comp.population)) from $S to $T"))
    return comp
end

function convert_component(comp::PopulationComponent{S, M, P}, ::Type{T}) where
        {S<:Real, M, P<:AbstractVector, T<:Real}
    PopulationComponent(comp.model, T.(comp.population);
        stage_names = comp.stage_names,
        species = comp.species,
        type = comp.type,
        patch = comp.patch)
end

# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

"""
    PopulationSystem(name => component, ...; state = NamedTuple())

Ordered collection of named [`PopulationComponent`](@ref)s plus named auxiliary state
exposed to rules, events and observables. A value that is not a component is wrapped with
`as_component`.

`state` is a `NamedTuple`, a dictionary or a vector of pairs, whose values are copied; or
a vector of [`AbstractStateVariable`](@ref)s, which are stored by reference under their
`name`.

Components are read with `system[name]`, iterated in order with `pairs(system)`, and
state with `get_state(system, name)`.
"""
struct PopulationSystem{T<:Real}
    order::Vector{Symbol}
    components::Dict{Symbol, PopulationComponent{T}}
    state::Dict{Symbol, Any}
end

function PopulationSystem(pairs::Pair{Symbol}...; state = NamedTuple())
    isempty(pairs) && throw(ArgumentError("PopulationSystem must have at least one component"))

    order = Symbol[]
    raw_components = PopulationComponent[]
    for (name, value) in pairs
        name in order && throw(ArgumentError("duplicate component name: $name"))
        comp = value isa PopulationComponent ? value : as_component(value)
        comp isa PopulationComponent || throw(ArgumentError(
            "expected PopulationComponent for :$name, got $(typeof(comp))"))
        push!(order, name)
        push!(raw_components, comp)
    end

    T = promote_type(map(_component_eltype, raw_components)...)
    components = Dict{Symbol, PopulationComponent{T}}()
    for (name, comp) in zip(order, raw_components)
        components[name] = convert_component(comp, T)
    end

    return PopulationSystem{T}(copy(order), components, _coerce_state(state))
end

function _coerce_state(state)
    if state isa NamedTuple
        return Dict{Symbol, Any}(k => deepcopy(v) for (k, v) in pairs(state))
    elseif state isa AbstractDict
        return Dict{Symbol, Any}(Symbol(k) => deepcopy(v) for (k, v) in pairs(state))
    elseif state isa AbstractVector && all(x -> x isa AbstractStateVariable, state)
        # State variables are shared with the caller, not copied.
        return Dict{Symbol, Any}(sv.name => sv for sv in state)
    elseif state isa AbstractVector && all(x -> x isa Pair, state)
        return Dict{Symbol, Any}(Symbol(k) => deepcopy(v) for (k, v) in state)
    else
        throw(ArgumentError(
            "state must be a NamedTuple, AbstractDict, vector of pairs, or vector of state variables"))
    end
end

Base.getindex(sys::PopulationSystem, name::Symbol) = sys.components[name]
Base.haskey(sys::PopulationSystem, name::Symbol) = haskey(sys.components, name)
Base.keys(sys::PopulationSystem) = sys.order
Base.length(sys::PopulationSystem) = length(sys.order)
Base.pairs(sys::PopulationSystem) = ((name, sys.components[name]) for name in sys.order)

function Base.show(io::IO, sys::PopulationSystem)
    state_suffix = isempty(sys.state) ? "" : ", $(length(sys.state)) state vars"
    print(io, "PopulationSystem($(length(sys)) components: $(join(sys.order, ", "))$state_suffix)")
end

# ---------------------------------------------------------------------------
# Auxiliary state
# ---------------------------------------------------------------------------

"""
    get_state(system, name)

Current value of the named auxiliary state of a [`PopulationSystem`](@ref). If the entry
is an [`AbstractStateVariable`](@ref) its value is returned (`get_state(variable)`),
otherwise the entry itself.
"""
function get_state(sys::PopulationSystem, name::Symbol)
    entry = sys.state[name]
    return entry isa AbstractStateVariable ? get_state(entry) : entry
end

function set_state!(sys::PopulationSystem, name::Symbol, value)
    if !(value isa AbstractStateVariable) && haskey(sys.state, name)
        entry = sys.state[name]
        entry isa AbstractStateVariable && return set_state!(entry, value)
    end
    return sys.state[name] = value
end

"""
    has_state(system, name)

Whether a [`PopulationSystem`](@ref) has auxiliary state called `name`.
"""
has_state(sys::PopulationSystem, name::Symbol) = haskey(sys.state, name)

# ---------------------------------------------------------------------------
# Totals and grouping
# ---------------------------------------------------------------------------

"""
    total_population(system)
    total_population(carrier)

Total number of individuals in a [`PopulationSystem`](@ref), summed over its components,
or in one population carrier. For a stage vector it is the sum of the entries.
"""
function total_population end

total_population(population::AbstractVector{<:Real}) = sum(population)
total_population(sys::PopulationSystem) =
    sum(total_population(comp.population) for (_, comp) in pairs(sys))

"""
    component_total(system, name)

Total number of individuals in one component.
"""
component_total(sys::PopulationSystem, name::Symbol) = total_population(sys[name].population)

"""
    component_totals(system)

Dictionary of the total number of individuals in each component.
"""
component_totals(sys::PopulationSystem{T}) where {T} =
    Dict{Symbol, T}(name => total_population(comp.population) for (name, comp) in pairs(sys))

"""
    by_species(system, species)

The `(name, component)` pairs of the components with the given species label.
"""
by_species(sys::PopulationSystem, species::Symbol) =
    [(name, comp) for (name, comp) in pairs(sys) if comp.species == species]

"""
    by_type(system, type)

The `(name, component)` pairs of the components with the given type label.
"""
by_type(sys::PopulationSystem, type::Symbol) =
    [(name, comp) for (name, comp) in pairs(sys) if comp.type == type]

"""
    by_patch(system, patch)

The `(name, component)` pairs of the components with the given patch label.
"""
by_patch(sys::PopulationSystem, patch::Symbol) =
    [(name, comp) for (name, comp) in pairs(sys) if comp.patch == patch]

# ---------------------------------------------------------------------------
# Adding and removing individuals
# ---------------------------------------------------------------------------

"""
    inject!(system, name, [stage,] amount)

Add `amount` individuals to a stage (the first by default) of the named component. The
work is done by `inject!(carrier, stage, amount)`. For a stage vector the amount must be
nonnegative and the stage in range.
"""
function inject! end

function inject!(sys::PopulationSystem, name::Symbol, stage::Int, amount::Real)
    inject!(sys[name].population, stage, amount)
    return nothing
end

inject!(sys::PopulationSystem, name::Symbol, amount::Real) = inject!(sys, name, 1, amount)

function inject!(population::AbstractVector{<:Real}, stage::Int, amount::Real)
    amount >= 0 || throw(ArgumentError("injected amount must be nonnegative"))
    _validate_stage(population, stage)
    population[stage] += amount
    return nothing
end

"""
    remove_fraction!(system, name, [stage,] fraction)

Remove a fraction of the individuals of the named component, from one stage or, without
`stage`, from all of them. The work is done by `remove_fraction!(carrier, [stage,]
fraction)`. For a stage vector the fraction must be finite and lie in `[0, 1]`.
"""
function remove_fraction! end

function remove_fraction!(sys::PopulationSystem, name::Symbol, fraction::Real)
    remove_fraction!(sys[name].population, fraction)
    return nothing
end

function remove_fraction!(sys::PopulationSystem, name::Symbol, stage::Int, fraction::Real)
    remove_fraction!(sys[name].population, stage, fraction)
    return nothing
end

function remove_fraction!(population::AbstractVector{<:Real}, fraction::Real)
    _validate_fraction("fraction", fraction)
    population .*= (one(eltype(population)) - eltype(population)(fraction))
    return nothing
end

function remove_fraction!(population::AbstractVector{<:Real}, stage::Int, fraction::Real)
    _validate_fraction("fraction", fraction)
    _validate_stage(population, stage)
    population[stage] *= (one(eltype(population)) - eltype(population)(fraction))
    return nothing
end

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

"""
    TransferRule(source, target, fraction_fn; source_stage = 1, target_stage = 1, name)
    TransferRule(source, source_stage, target, target_stage, fraction_fn; name)

Move a fraction of one stage of `source` into a stage of `target` after stepping.
`fraction_fn` returns the fraction; its arguments are those the solving package passes to
rule functions.
"""
struct TransferRule{F} <: AbstractInteractionRule
    name::Symbol
    source::Symbol
    source_stage::Int
    target::Symbol
    target_stage::Int
    fraction_fn::F
end

function TransferRule(source::Symbol, target::Symbol, fraction_fn;
        source_stage::Int = 1,
        target_stage::Int = 1,
        name::Symbol = Symbol(source, "_to_", target))
    return TransferRule(name, source, source_stage, target, target_stage, fraction_fn)
end

function TransferRule(source::Symbol, source_stage::Int, target::Symbol, target_stage::Int,
        fraction_fn; name::Symbol = Symbol(source, "_to_", target))
    return TransferRule(name, source, source_stage, target, target_stage, fraction_fn)
end

"""
    ReproductionRule(target, reproduction_fn; stage = 1, name)
    ReproductionRule(target, stage, reproduction_fn; name)

Add offspring to a stage of `target` after stepping. `reproduction_fn` returns the number
of offspring.
"""
struct ReproductionRule{F} <: AbstractInteractionRule
    name::Symbol
    target::Symbol
    stage::Int
    reproduction_fn::F
end

function ReproductionRule(target::Symbol, reproduction_fn;
        stage::Int = 1,
        name::Symbol = Symbol("reproduce_", target))
    stage >= 1 || throw(ArgumentError("stage must be >= 1"))
    return ReproductionRule(name, target, stage, reproduction_fn)
end

ReproductionRule(target::Symbol, stage::Int, reproduction_fn;
        name::Symbol = Symbol("reproduce_", target)) =
    ReproductionRule(target, reproduction_fn; stage = stage, name = name)

"""
    MortalityRule(target, mortality_fn; stage = 0, name)

Apply extra mortality to one stage of `target` after stepping, or to all stages
(`stage = 0`). `mortality_fn` returns the fraction removed.
"""
struct MortalityRule{F} <: AbstractInteractionRule
    name::Symbol
    target::Symbol
    stage::Int
    mortality_fn::F
end

function MortalityRule(target::Symbol, mortality_fn;
        stage::Int = 0,
        name::Symbol = Symbol("mortality_", target))
    stage >= 0 || throw(ArgumentError("stage must be >= 0"))
    return MortalityRule(name, target, stage, mortality_fn)
end

"""
    CustomRule(name, apply_fn)

User-defined rule. What `apply_fn` receives and must return is set by the solving package.
"""
struct CustomRule{F} <: AbstractInteractionRule
    name::Symbol
    apply_fn::F
end

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

"""
    apply_event!(event, system, day, p) -> Bool
    apply_event!(event, system, environment, day, p) -> Bool

Apply a scheduled event to `system` if it is due on `day`, and return whether it fired.
The second form is for solvers that pass the day's environment (weather) to events; for
an event that does not use it, it falls back to the first.
"""
function apply_event! end

apply_event!(ev::AbstractScheduledEvent, sys, env, day::Int, p) = apply_event!(ev, sys, day, p)

"""
    PulseRelease(target, amount, interval; stage_idx = 1, start_day = 0, end_day, name)
    PulseRelease(target, stage_idx, amount, interval; start_day = 0, end_day, name)

Add a fixed amount to a stage of `target` every `interval` days, on `start_day`,
`start_day + interval`, … up to `end_day`.
"""
struct PulseRelease{T<:Real} <: AbstractScheduledEvent
    name::Symbol
    target::Symbol
    stage_idx::Int
    amount::T
    interval::Int
    start_day::Int
    end_day::Int
end

function PulseRelease(target::Symbol, amount::Real, interval::Int;
        stage_idx::Int = 1,
        start_day::Int = 0,
        end_day::Int = typemax(Int),
        name::Symbol = Symbol("pulse_", target))
    interval > 0 || throw(ArgumentError("interval must be positive"))
    stage_idx >= 1 || throw(ArgumentError("stage_idx must be >= 1"))
    _validate_nonnegative("release amount", amount)
    return PulseRelease(name, target, stage_idx, float(amount), interval, start_day, end_day)
end

PulseRelease(target::Symbol, stage_idx::Int, amount::Real, interval::Int; kwargs...) =
    PulseRelease(target, amount, interval; stage_idx = stage_idx, kwargs...)

function apply_event!(ev::PulseRelease, sys::PopulationSystem, day::Int, p)
    day < ev.start_day && return false
    day > ev.end_day && return false
    (day - ev.start_day) % ev.interval == 0 || return false
    inject!(sys, ev.target, ev.stage_idx, ev.amount)
    return true
end

"""
    SingleDayRelease(target, amount, day; stage_idx = 1, name)
    SingleDayRelease(target, stage_idx, amount, day; name)

Add a fixed amount to a stage of `target` on exactly one day.
"""
struct SingleDayRelease{T<:Real} <: AbstractScheduledEvent
    name::Symbol
    target::Symbol
    stage_idx::Int
    amount::T
    day::Int
end

function SingleDayRelease(target::Symbol, amount::Real, day::Int;
        stage_idx::Int = 1,
        name::Symbol = Symbol("release_", target))
    stage_idx >= 1 || throw(ArgumentError("stage_idx must be >= 1"))
    _validate_nonnegative("release amount", amount)
    return SingleDayRelease(name, target, stage_idx, float(amount), day)
end

SingleDayRelease(target::Symbol, stage_idx::Int, amount::Real, day::Int; kwargs...) =
    SingleDayRelease(target, amount, day; stage_idx = stage_idx, kwargs...)

function apply_event!(ev::SingleDayRelease, sys::PopulationSystem, day::Int, p)
    day == ev.day || return false
    inject!(sys, ev.target, ev.stage_idx, ev.amount)
    return true
end

"""
    SprayEvent(targets, kill_fractions, days; name = :spray)

Remove a fraction of each named component on the given days. `targets` and
`kill_fractions` are parallel vectors. A target that is not a component of the system is
an error when the event fires.
"""
struct SprayEvent <: AbstractScheduledEvent
    name::Symbol
    targets::Vector{Symbol}
    kill_fractions::Vector{Float64}
    days::Vector{Int}
end

function SprayEvent(targets::AbstractVector{Symbol},
        kill_fractions::AbstractVector{<:Real},
        days::AbstractVector{Int};
        name::Symbol = :spray)
    length(targets) == length(kill_fractions) || throw(DimensionMismatch(
        "targets length $(length(targets)) does not match kill_fractions length $(length(kill_fractions))"))
    foreach(frac -> _validate_fraction("kill fraction", frac), kill_fractions)
    return SprayEvent(name, collect(targets), Float64.(kill_fractions), collect(days))
end

function apply_event!(ev::SprayEvent, sys::PopulationSystem, day::Int, p)
    day in ev.days || return false
    for (target, frac) in zip(ev.targets, ev.kill_fractions)
        haskey(sys, target) || throw(ArgumentError("unknown spray target :$target"))
        remove_fraction!(sys, target, frac)
    end
    return true
end

# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

"""
    Observable(name, fn)

A named quantity recorded at each stored time point. What `fn` receives is set by the
solving package.
"""
struct Observable{F}
    name::Symbol
    fn::F
end
