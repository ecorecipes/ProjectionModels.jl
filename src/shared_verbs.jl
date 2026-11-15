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
