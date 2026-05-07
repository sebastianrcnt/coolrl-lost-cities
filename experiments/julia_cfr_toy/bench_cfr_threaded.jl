using Printf
using Statistics
using Base.Threads

include("bench_cfr.jl")

const THREAD_GRID = (1, 2, 4, 8)
const RUNS_PER_CASE = 5
const ROOT_ACTIONS = 4

function iteration_ranges(chunks::Int)::Vector{UnitRange{Int}}
    ranges = Vector{UnitRange{Int}}(undef, chunks)
    base = MEASURE_ITERATIONS ÷ chunks
    extra = MEASURE_ITERATIONS % chunks
    start = 1
    for chunk in 1:chunks
        width = base + (chunk <= extra ? 1 : 0)
        stop = start + width - 1
        ranges[chunk] = start:stop
        start = stop + 1
    end
    return ranges
end

function reset_trees!(trees::Vector{CFRTree}, chunks::Int)
    for chunk in 1:chunks
        fill!(trees[chunk].regret, 0.0)
    end
end

function reduce_root!(out::Vector{Float64}, trees::Vector{CFRTree}, chunks::Int)
    fill!(out, 0.0)
    @inbounds for chunk in 1:chunks
        tree = trees[chunk]
        out[1] += tree.regret[regret_index(1, 1)]
        out[2] += tree.regret[regret_index(1, 2)]
        out[3] += tree.regret[regret_index(1, 3)]
        out[4] += tree.regret[regret_index(1, 4)]
    end
end

function run_chunk!(tree::CFRTree, range::UnitRange{Int})
    value = 0.0
    for iteration in range
        value += run_iteration!(tree, iteration)
    end
    return value
end

function run_case!(
    out::Vector{Float64},
    trees::Vector{CFRTree},
    ranges::Vector{UnitRange{Int}},
    chunks::Int;
    threaded::Bool,
)
    reset_trees!(trees, chunks)
    if threaded && chunks > 1
        @threads for chunk in 1:chunks
            run_chunk!(trees[chunk], ranges[chunk])
        end
    else
        for chunk in 1:chunks
            run_chunk!(trees[chunk], ranges[chunk])
        end
    end
    reduce_root!(out, trees, chunks)
end

function assert_close(label::AbstractString, actual::Vector{Float64}, expected::Vector{Float64})
    max_diff = maximum(abs.(actual .- expected))
    if max_diff > 1e-9
        error("$label parity failed: max_diff=$max_diff actual=$actual expected=$expected")
    end
end

function measure_case(chunks::Int)
    ranges = iteration_ranges(chunks)
    reference_trees = [CFRTree() for _ in 1:chunks]
    measured_trees = [CFRTree() for _ in 1:chunks]
    reference = zeros(Float64, ROOT_ACTIONS)
    actual = zeros(Float64, ROOT_ACTIONS)

    run_case!(reference, reference_trees, ranges, chunks; threaded=false)
    run_case!(actual, measured_trees, ranges, chunks; threaded=chunks > 1)
    assert_close("warmup $(chunks)T", actual, reference)

    times = Vector{Float64}(undef, RUNS_PER_CASE)
    allocs = Vector{Int}(undef, RUNS_PER_CASE)
    gc_times = Vector{Float64}(undef, RUNS_PER_CASE)
    for run in 1:RUNS_PER_CASE
        GC.gc()
        before = Base.gc_num()
        elapsed_ref = Ref(0.0)
        allocated = @allocated begin
            elapsed_ref[] = @elapsed begin
                run_case!(actual, measured_trees, ranges, chunks; threaded=chunks > 1)
            end
        end
        after = Base.gc_num()
        assert_close("run $(run) $(chunks)T", actual, reference)
        times[run] = elapsed_ref[]
        allocs[run] = allocated
        gc_times[run] = (after.total_time - before.total_time) / 1e9
    end

    total_s = median(times)
    alloc_mb = median(allocs) / 1024.0 / 1024.0
    gc_time_s = median(gc_times)
    return Dict(
        "threads" => chunks,
        "total_s" => total_s,
        "iter_ms" => total_s * 1000.0 / MEASURE_ITERATIONS,
        "traversal_us" => total_s * 1_000_000.0 / (MEASURE_ITERATIONS * TRAVERSALS_PER_ITER),
        "alloc_mb" => alloc_mb,
        "gc_time_s" => gc_time_s,
        "gc_share" => total_s > 0.0 ? gc_time_s / total_s : 0.0,
        "root_regret" => copy(reference),
    )
end

function scaling_label(efficiency::Float64)::String
    if efficiency >= 0.80
        return "near-linear"
    elseif efficiency >= 0.50
        return "partial scale"
    end
    return "contention"
end

function main()
    available = Threads.nthreads()
    grid = [threads for threads in THREAD_GRID if threads <= available]
    if isempty(grid)
        error("no thread counts available")
    end

    results = [measure_case(threads) for threads in grid]
    one_thread_s = results[1]["total_s"]

    println("threads  iter_ms  total_s  μs/trav  alloc_MB  gc_s  gc_share  speedup  efficiency")
    for result in results
        threads = result["threads"]
        speedup = one_thread_s / result["total_s"]
        efficiency = speedup / threads
        @printf(
            "%-7d %8.3f %8.4f %8.3f %9.3f %5.3f %8.1f%% %8.2f× %9.0f%%\n",
            threads,
            result["iter_ms"],
            result["total_s"],
            result["traversal_us"],
            result["alloc_mb"],
            result["gc_time_s"],
            result["gc_share"] * 100.0,
            speedup,
            efficiency * 100.0,
        )
    end

    last = results[end]
    speedup = one_thread_s / last["total_s"]
    efficiency = speedup / last["threads"]
    @printf("%dT efficiency %.0f%% — %s\n", last["threads"], efficiency * 100.0, scaling_label(efficiency))
end

main()
