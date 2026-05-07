using JSON

include("../src/SafeHeuristic.jl")
using .SafeHeuristic

function main()
    if length(ARGS) != 1
        println(stderr, "usage: julia --project=experiments/julia_safe_heuristic experiments/julia_safe_heuristic/test/parity.jl <snapshots.jsonl>")
        exit(2)
    end
    path = ARGS[1]
    checked = 0
    for line in eachline(path)
        isempty(strip(line)) && continue
        record = JSON.parse(line)
        actual = safe_heuristic_action(record)
        expected = Int(record["expected_action"])
        if actual != expected
            println(stderr, "mismatch config=$(record["config_name"]) variant=$(record["variant"]) seed=$(record["seed"]) turn=$(record["turn"]) phase=$(record["phase"]) player=$(record["current_player"]) expected=$expected actual=$actual")
            exit(1)
        end
        checked += 1
    end
    println("checked $checked snapshots")
end

main()
