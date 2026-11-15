"""
Functions that several model packages implement, each for its own problem type.

They are declared here, without methods, so that every package adds methods to the same
function instead of defining a separate function of the same name. The names can then be
used unqualified whichever packages are loaded together. (`remake` is shared in the same
way, through `SciMLBase.remake`.)
"""

"""
    demographic_ensemble(prob; n_reps = 100, rng = Random.default_rng(), ...)

Run `n_reps` independent realizations of `prob` with demographic stochasticity and return
`(totals, sols)`: `totals` is an `(n_time × n_reps)` matrix of total population sizes,
directly consumable by [`quasi_extinction`](@ref), and `sols` the individual solutions.

Each model package adds the method for its own problem type (`MPMProblem`, `IPMProblem`,
continuous-state and finite-state problems).
"""
function demographic_ensemble end

"""
    to_discrete_problem(prob)

Convert a discrete-time projection problem to a `SciMLBase.DiscreteProblem`, for use with
discrete solvers from the SciML ecosystem.

Each model package adds the method for its own problem type (`MPMProblem`,
`CoupledMPMProblem`, `IPMProblem`).
"""
function to_discrete_problem end

"""
    to_ode_problem(prob)

Convert a continuous-time problem to a `SciMLBase.ODEProblem`.

Each package adds the method for its own problem type (continuous-state generator and
transport problems, finite-state generator problems).
"""
function to_ode_problem end

"""
    to_dde_problem(prob)

Convert a continuous-time problem with delays to a `SciMLBase.DDEProblem`.

Each package adds the method for its own problem type (continuous-state and finite-state
problems with delay terms).
"""
function to_dde_problem end

"""
    set_state!(system, name, value)
    set_state!(...)

Set a piece of the state that a model carries.

`set_state!(system, name, value)` sets the named auxiliary state of a
[`PopulationSystem`](@ref StructuredPopulationCore.PopulationSystem). If the entry is an
[`AbstractStateVariable`](@ref StructuredPopulationCore.AbstractStateVariable) the call is
passed on to `set_state!(variable, value)`; otherwise the entry is replaced, or created.
FiniteStatePopulationDynamics.jl adds `set_state!(integrator, domain, label, value)` for
use in events.

Declared here so that these are one function. It is not exported from the core package:
the packages that use it export it.
"""
function set_state! end
