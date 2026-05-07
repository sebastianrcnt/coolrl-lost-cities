using JSON

function main()
    try
        @eval using Torch
    catch err
        result = Dict(
            "backend" => "Torch.jl",
            "status" => "BLOCKED",
            "error" => sprint(showerror, err),
        )
        println(JSON.json(result))
        exit(1)
    end
end

main()
