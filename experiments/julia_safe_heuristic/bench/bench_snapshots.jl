using BenchmarkTools
using JSON

include("../src/SafeHeuristic.jl")
using .SafeHeuristic

function main()
    if length(ARGS) != 1
        println(stderr, "usage: julia --project=experiments/julia_safe_heuristic experiments/julia_safe_heuristic/bench/bench_snapshots.jl <snapshots.jsonl>")
        exit(2)
    end
    records = [JSON.parse(line) for line in eachline(ARGS[1]) if !isempty(strip(line))]
    println("loaded $(length(records)) snapshots")
    for record in records[1:min(end, 100)]
        safe_heuristic_action(record)
    end
    result = @benchmark begin
        total = 0
        for record in $records
            total += safe_heuristic_action(record)
        end
        total
    end
    display(result)
end

main()
