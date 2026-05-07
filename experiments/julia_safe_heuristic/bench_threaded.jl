import Pkg

const PROJECT_DIR = @__DIR__
Pkg.activate(PROJECT_DIR; io=devnull)

using JSON
using Printf
using Statistics
using Base.Threads

include(joinpath(PROJECT_DIR, "src", "SafeHeuristic.jl"))
using .SafeHeuristic

const DEFAULT_SNAPSHOT_PATH = joinpath(
    dirname(dirname(PROJECT_DIR)),
    "runs",
    "tmp",
    "safe_heuristic_snapshots_3seeds.jsonl",
)
const RUNS_PER_CASE = 5

function load_records(path::AbstractString)
    return [JSON.parse(line) for line in eachline(path) if !isempty(strip(line))]
end

function chunk_ranges(length::Int, chunks::Int)::Vector{UnitRange{Int}}
    ranges = Vector{UnitRange{Int}}(undef, chunks)
    base = length ÷ chunks
    extra = length % chunks
    start = 1
    for chunk in 1:chunks
        width = base + (chunk <= extra ? 1 : 0)
        stop = start + width - 1
        ranges[chunk] = start:stop
        start = stop + 1
    end
    return ranges
end

function run_sequential(records)::Vector{Int}
    actions = Vector{Int}(undef, length(records))
    @inbounds for idx in eachindex(records)
        actions[idx] = safe_heuristic_action(records[idx])
    end
    return actions
end

function run_threaded(records, chunks::Int)::Vector{Int}
    actions = Vector{Int}(undef, length(records))
    ranges = chunk_ranges(length(records), chunks)
    @threads for chunk in eachindex(ranges)
        @inbounds for idx in ranges[chunk]
            actions[idx] = safe_heuristic_action(records[idx])
        end
    end
    return actions
end

function run_case(records, chunks::Int)::Vector{Int}
    if chunks == 1
        return run_sequential(records)
    end
    return run_threaded(records, chunks)
end

function measure_case(records, chunks::Int, baseline::Vector{Int})::Float64
    warmup_actions = run_case(records, chunks)
    if warmup_actions != baseline
        error("action parity failed during warmup for $(chunks) thread case")
    end

    times = Vector{Float64}(undef, RUNS_PER_CASE)
    for run in 1:RUNS_PER_CASE
        actions = Vector{Int}()
        elapsed = @elapsed begin
            actions = run_case(records, chunks)
        end
        if actions != baseline
            error("action parity failed on run $(run) for $(chunks) thread case")
        end
        times[run] = elapsed * 1000.0
    end
    return median(times)
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
    if length(ARGS) > 1
        println(stderr, "usage: julia --threads=8 experiments/julia_safe_heuristic/bench_threaded.jl [snapshots.jsonl]")
        exit(2)
    end

    path = length(ARGS) == 1 ? ARGS[1] : DEFAULT_SNAPSHOT_PATH
    if !isfile(path)
        println(stderr, "snapshot file not found: $path")
        exit(2)
    end

    records = load_records(path)
    available = Threads.nthreads()
    grid = [threads for threads in (1, 2, 4, 8) if threads <= available && threads <= length(records)]
    if isempty(grid)
        println(stderr, "no thread counts are available")
        exit(2)
    end

    baseline = run_sequential(records)
    results = Dict{Int,Float64}()
    for threads in grid
        results[threads] = measure_case(records, threads, baseline)
    end

    one_thread_ms = results[1]
    println("threads  total_ms  μs/call  speedup vs 1T  efficiency")
    for threads in grid
        total_ms = results[threads]
        us_per_call = total_ms * 1000.0 / length(records)
        speedup = one_thread_ms / total_ms
        efficiency = speedup / threads
        @printf("%-7d %8.1f %8.1f %13.2f× %10.0f%%\n", threads, total_ms, us_per_call, speedup, efficiency * 100.0)
    end

    if 8 in grid
        speedup = one_thread_ms / results[8]
        efficiency = speedup / 8
        @printf("8T efficiency %.0f%% — %s\n", efficiency * 100.0, scaling_label(efficiency))
    end
end

main()
