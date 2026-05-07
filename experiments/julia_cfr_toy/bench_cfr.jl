using Printf

const MAX_DEPTH = 10
const BRANCHING = 4
const HALF_DEPTH = MAX_DEPTH ÷ 2
const TRAVERSALS_PER_ITER = 1000
const MEASURE_ITERATIONS = 100
const SEED = UInt64(0x00000000013579bd)
const NUM_INTERNAL_NODES = (BRANCHING^MAX_DEPTH - 1) ÷ (BRANCHING - 1)
const MASK64 = typemax(UInt64)
const TWO_POW_53 = 9007199254740992.0

mutable struct CFRTree
    regret::Vector{Float64}
end

function CFRTree()
    return CFRTree(zeros(Float64, NUM_INTERNAL_NODES * BRANCHING))
end

@inline function regret_index(node_id::Int, action::Int)::Int
    return (node_id - 1) * BRANCHING + action
end

@inline function splitmix64(x::UInt64)::UInt64
    x += UInt64(0x9e3779b97f4a7c15)
    x = (x ⊻ (x >> 30)) * UInt64(0xbf58476d1ce4e5b9)
    x = (x ⊻ (x >> 27)) * UInt64(0x94d049bb133111eb)
    return x ⊻ (x >> 31)
end

@inline function hash_key(node_id::Int, depth::Int, action::Int, traversal::Int)::UInt64
    x = SEED
    x ⊻= UInt64(node_id) * UInt64(0xd6e8feb86659fd93)
    x ⊻= UInt64(depth + 1) * UInt64(0xa5a3564e27f886d9)
    x ⊻= UInt64(action + 11) * UInt64(0x9e3779b185ebca87)
    x ⊻= UInt64(traversal + 17) * UInt64(0xc2b2ae3d27d4eb4f)
    return splitmix64(x)
end

@inline function unit_value(key::UInt64)::Float64
    bits = key >> 11
    return Float64(bits) / TWO_POW_53
end

@inline function terminal_value(node_id::Int, depth::Int, action::Int, traversal::Int)::Float64
    return 2.0 * unit_value(hash_key(node_id, depth, action, traversal)) - 1.0
end

@inline function is_legal(depth::Int, action::Int)::Bool
    return depth < HALF_DEPTH || action != BRANCHING
end

function traverse!(tree::CFRTree, node_id::Int, depth::Int, traversal::Int)::Float64
    positive_sum = 0.0
    legal_count = 0
    strategy1 = 0.0
    strategy2 = 0.0
    strategy3 = 0.0
    strategy4 = 0.0
    cf1 = 0.0
    cf2 = 0.0
    cf3 = 0.0
    cf4 = 0.0

    @inbounds for action in 1:BRANCHING
        if is_legal(depth, action)
            legal_count += 1
            positive = max(tree.regret[regret_index(node_id, action)], 0.0)
            positive_sum += positive
        end
    end

    if positive_sum > 0.0
        @inbounds begin
            strategy1 = is_legal(depth, 1) ? max(tree.regret[regret_index(node_id, 1)], 0.0) / positive_sum : 0.0
            strategy2 = is_legal(depth, 2) ? max(tree.regret[regret_index(node_id, 2)], 0.0) / positive_sum : 0.0
            strategy3 = is_legal(depth, 3) ? max(tree.regret[regret_index(node_id, 3)], 0.0) / positive_sum : 0.0
            strategy4 = is_legal(depth, 4) ? max(tree.regret[regret_index(node_id, 4)], 0.0) / positive_sum : 0.0
        end
    else
        uniform = 1.0 / legal_count
        strategy1 = is_legal(depth, 1) ? uniform : 0.0
        strategy2 = is_legal(depth, 2) ? uniform : 0.0
        strategy3 = is_legal(depth, 3) ? uniform : 0.0
        strategy4 = is_legal(depth, 4) ? uniform : 0.0
    end

    r = unit_value(hash_key(node_id, depth, 97, traversal))
    cumulative = 0.0
    sampled_action = 1
    cumulative += strategy1
    if r <= cumulative
        sampled_action = 1
    else
        cumulative += strategy2
        if r <= cumulative
            sampled_action = 2
        else
            cumulative += strategy3
            if r <= cumulative
                sampled_action = 3
            else
                sampled_action = 4
            end
        end
    end

    sampled_value = 0.0
    child_id = (node_id - 1) * BRANCHING + sampled_action + 1
    if depth + 1 >= MAX_DEPTH
        sampled_value = terminal_value(node_id, depth, sampled_action, traversal)
    else
        sampled_value = traverse!(tree, child_id, depth + 1, traversal)
    end

    cf1 = if !is_legal(depth, 1)
        0.0
    elseif sampled_action == 1
        sampled_value
    else
        terminal_value(node_id, depth, 1, traversal)
    end
    cf2 = if !is_legal(depth, 2)
        0.0
    elseif sampled_action == 2
        sampled_value
    else
        terminal_value(node_id, depth, 2, traversal)
    end
    cf3 = if !is_legal(depth, 3)
        0.0
    elseif sampled_action == 3
        sampled_value
    else
        terminal_value(node_id, depth, 3, traversal)
    end
    cf4 = if !is_legal(depth, 4)
        0.0
    elseif sampled_action == 4
        sampled_value
    else
        terminal_value(node_id, depth, 4, traversal)
    end

    @inbounds begin
        if is_legal(depth, 1)
            tree.regret[regret_index(node_id, 1)] += cf1 - sampled_value
        end
        if is_legal(depth, 2)
            tree.regret[regret_index(node_id, 2)] += cf2 - sampled_value
        end
        if is_legal(depth, 3)
            tree.regret[regret_index(node_id, 3)] += cf3 - sampled_value
        end
        if is_legal(depth, 4)
            tree.regret[regret_index(node_id, 4)] += cf4 - sampled_value
        end
    end

    return strategy1 * cf1 + strategy2 * cf2 + strategy3 * cf3 + strategy4 * cf4
end

function run_iteration!(tree::CFRTree, iteration::Int)
    base = (iteration - 1) * TRAVERSALS_PER_ITER
    value = 0.0
    for offset in 1:TRAVERSALS_PER_ITER
        value += traverse!(tree, 1, 0, base + offset)
    end
    return value
end

function run_benchmark()
    warmup_tree = CFRTree()
    run_iteration!(warmup_tree, 1)

    GC.gc()
    tree = CFRTree()
    gc_before = Base.gc_num()
    elapsed_ref = Ref(0.0)
    allocated = @allocated begin
        elapsed_ref[] = @elapsed begin
            for iteration in 1:MEASURE_ITERATIONS
                run_iteration!(tree, iteration)
            end
        end
    end
    gc_after = Base.gc_num()
    gc_time = (gc_after.total_time - gc_before.total_time) / 1e9
    total = elapsed_ref[]
    alloc_mb = allocated / 1024.0 / 1024.0
    iter_ms = total * 1000.0 / MEASURE_ITERATIONS
    traversal_us = total * 1_000_000.0 / (MEASURE_ITERATIONS * TRAVERSALS_PER_ITER)
    root_regret = [
        tree.regret[regret_index(1, 1)],
        tree.regret[regret_index(1, 2)],
        tree.regret[regret_index(1, 3)],
        tree.regret[regret_index(1, 4)],
    ]
    return Dict(
        "lang" => "Julia",
        "iterations" => MEASURE_ITERATIONS,
        "traversals_per_iter" => TRAVERSALS_PER_ITER,
        "total_s" => total,
        "iter_ms" => iter_ms,
        "traversal_us" => traversal_us,
        "alloc_mb" => alloc_mb,
        "gc_time_s" => gc_time,
        "gc_share" => total > 0.0 ? gc_time / total : 0.0,
        "root_regret" => root_regret,
    )
end

function print_json(result)
    @printf(
        "{\"lang\":\"Julia\",\"iterations\":%d,\"traversals_per_iter\":%d,\"total_s\":%.17g,\"iter_ms\":%.17g,\"traversal_us\":%.17g,\"alloc_mb\":%.17g,\"gc_time_s\":%.17g,\"gc_share\":%.17g,\"root_regret\":[%.17g,%.17g,%.17g,%.17g]}\n",
        result["iterations"],
        result["traversals_per_iter"],
        result["total_s"],
        result["iter_ms"],
        result["traversal_us"],
        result["alloc_mb"],
        result["gc_time_s"],
        result["gc_share"],
        result["root_regret"]...,
    )
end

function print_table(result)
    println("Lang     iter mean (ms)  total (s)  alloc (MB)  gc time (s)  gc share")
    @printf(
        "%-8s %14.2f %10.2f %11.1f %12.2f %8.1f%%\n",
        result["lang"],
        result["iter_ms"],
        result["total_s"],
        result["alloc_mb"],
        result["gc_time_s"],
        result["gc_share"] * 100.0,
    )
    @printf("mean traversal: %.2f μs\n", result["traversal_us"])
    @printf("root regret: [%.12f, %.12f, %.12f, %.12f]\n", result["root_regret"]...)
end

function main()
    result = run_benchmark()
    if "--json" in ARGS
        print_json(result)
    else
        print_table(result)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
