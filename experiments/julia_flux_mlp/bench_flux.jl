using CUDA
using Flux
using JSON
using Printf

const WARMUP_ITERS = 10
const TIMED_ITERS = 100

function dense_from_payload(layer)
    weight = Float32.(layer["weight"])
    bias = Float32.(layer["bias"])
    out_dim = Int(layer["out_dim"])
    in_dim = Int(layer["in_dim"])
    dense = Dense(in_dim => out_dim)
    dense.weight .= reshape(weight, in_dim, out_dim)'
    dense.bias .= bias
    return dense
end

function build_model(payload)
    layers = Any[]
    dense_layers = payload["layers"]
    for (index, layer) in enumerate(dense_layers)
        push!(layers, dense_from_payload(layer))
        if index < length(dense_layers)
            push!(layers, relu)
        end
    end
    return Flux.fmap(cu, Chain(layers...))
end

function input_matrix(batch_payload)
    batch_size = Int(batch_payload["batch_size"])
    input_dim = Int(batch_payload["input_dim"])
    flat = Float32.(batch_payload["input"])
    x = Matrix{Float32}(undef, input_dim, batch_size)
    @inbounds for sample in 1:batch_size
        source_base = (sample - 1) * input_dim
        for feature in 1:input_dim
            x[feature, sample] = flat[source_base + feature]
        end
    end
    return cu(x)
end

function flatten_output(y)
    cpu = Array(y)
    out_dim, batch_size = size(cpu)
    flat = Vector{Float32}(undef, batch_size * out_dim)
    @inbounds for sample in 1:batch_size
        dest_base = (sample - 1) * out_dim
        for output in 1:out_dim
            flat[dest_base + output] = cpu[output, sample]
        end
    end
    return flat
end

function timed_forward_ms(model, x)
    CUDA.synchronize()
    for _ in 1:WARMUP_ITERS
        model(x)
    end
    CUDA.synchronize()

    elapsed = @elapsed begin
        for _ in 1:TIMED_ITERS
            model(x)
        end
        CUDA.synchronize()
    end
    return elapsed * 1000.0 / TIMED_ITERS
end

function benchmark(payload)
    CUDA.allowscalar(false)
    model = build_model(payload)
    results = Dict{String,Any}()
    max_abs_diff = 0.0
    for batch_payload in payload["batches"]
        batch_size = Int(batch_payload["batch_size"])
        x = input_matrix(batch_payload)
        y = model(x)
        CUDA.synchronize()
        actual = flatten_output(y)
        expected = Float32.(batch_payload["expected_output"])
        diff = maximum(abs.(actual .- expected))
        max_abs_diff = max(max_abs_diff, Float64(diff))
        ms = timed_forward_ms(model, x)
        results[string(batch_size)] = Dict(
            "forward_ms" => ms,
            "us_per_state" => ms * 1000.0 / batch_size,
            "max_abs_diff" => Float64(diff),
        )
    end
    return Dict(
        "lang" => "Julia/Flux",
        "device" => string(CUDA.name(CUDA.device())),
        "timed_iters" => TIMED_ITERS,
        "warmup_iters" => WARMUP_ITERS,
        "max_abs_diff" => max_abs_diff,
        "batches" => results,
    )
end

function print_json(result)
    println(JSON.json(result))
end

function main()
    if length(ARGS) != 1
        println(stderr, "usage: julia --project=experiments/julia_flux_mlp experiments/julia_flux_mlp/bench_flux.jl <payload.json>")
        exit(2)
    end
    payload = JSON.parsefile(ARGS[1])
    result = benchmark(payload)
    print_json(result)
end

main()
