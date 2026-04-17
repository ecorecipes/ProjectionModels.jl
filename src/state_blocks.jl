"""
Named contiguous block layouts for flattened state vectors.

These are useful when several related projection-model components share one
vector state but still need stable named ranges for splitting and recombining
that state.
"""

"""
    StateBlockLayout(names, sizes)

Describe a flattened state vector made of named contiguous blocks.

`names[i]` names the block occupying `sizes[i]` consecutive entries.
"""
struct StateBlockLayout
    names::Vector{Symbol}
    ranges::Dict{Symbol, UnitRange{Int}}
    n_states::Int
end

function StateBlockLayout(names::AbstractVector{Symbol}, sizes::AbstractVector{<:Integer})
    length(names) == length(sizes) || throw(DimensionMismatch(
        "names has length $(length(names)); sizes has length $(length(sizes))"))
    isempty(names) && throw(ArgumentError("StateBlockLayout must contain at least one block"))

    order = Symbol[]
    ranges = Dict{Symbol, UnitRange{Int}}()
    start = 1

    for (name, size) in zip(names, sizes)
        name in order && throw(ArgumentError("duplicate block name: $name"))
        size > 0 || throw(ArgumentError("block size for :$name must be positive, got $size"))
        stop = start + Int(size) - 1
        push!(order, name)
        ranges[name] = start:stop
        start = stop + 1
    end

    return StateBlockLayout(order, ranges, start - 1)
end

function StateBlockLayout(blocks::Pair{Symbol, <:Integer}...)
    return StateBlockLayout(collect(first.(blocks)), [last(block) for block in blocks])
end

Base.length(layout::StateBlockLayout) = layout.n_states
Base.getindex(layout::StateBlockLayout, name::Symbol) = layout.ranges[name]
Base.keys(layout::StateBlockLayout) = layout.names

function Base.show(io::IO, layout::StateBlockLayout)
    print(io, "StateBlockLayout(")
    for (idx, name) in enumerate(layout.names)
        idx > 1 && print(io, ", ")
        print(io, name, "=>", length(layout[name]))
    end
    print(io, ")")
end

"""
    blocknames(layout)

Return the block names in layout order.
"""
blocknames(layout::StateBlockLayout) = layout.names

"""
    blockrange(layout, name)

Return the contiguous index range for block `name`.
"""
blockrange(layout::StateBlockLayout, name::Symbol) = layout[name]

"""
    blockranges(layout)

Return a dictionary of block ranges keyed by name.
"""
blockranges(layout::StateBlockLayout) =
    Dict{Symbol, UnitRange{Int}}(name => layout[name] for name in layout.names)

"""
    split_state(state, layout; copy=true)

Split a flattened state vector into named blocks.

When `copy=false`, the returned dictionary contains views into `state`.
"""
function split_state(state::AbstractVector, layout::StateBlockLayout; copy::Bool=true)
    length(state) == length(layout) || throw(DimensionMismatch(
        "state has length $(length(state)); layout expects $(length(layout))"))

    if copy
        T = eltype(state)
        return Dict{Symbol, Vector{T}}(
            name => collect(state[layout[name]]) for name in layout.names)
    end

    return Dict{Symbol, AbstractVector}(
        name => view(state, layout[name]) for name in layout.names)
end

"""
    combine_state(layout, blocks)

Combine named state blocks back into one flattened state vector.
"""
function combine_state(layout::StateBlockLayout, blocks)
    Ts = Type[]
    for name in layout.names
        haskey(blocks, name) || throw(KeyError(name))
        push!(Ts, eltype(blocks[name]))
    end
    T = isempty(Ts) ? Float64 : promote_type(Ts...)
    state = Vector{T}(undef, length(layout))

    for name in layout.names
        block = blocks[name]
        range = layout[name]
        length(block) == length(range) || throw(DimensionMismatch(
            "block :$name has length $(length(block)); expected $(length(range))"))
        state[range] = T.(block)
    end

    return state
end
