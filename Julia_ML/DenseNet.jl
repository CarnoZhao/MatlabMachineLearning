using Flux
using CUDAdrv, CuArrays
using Flux: onehotbatch, crossentropy, onecold, glorot_normal, throttle
using Statistics
using Base.Iterators: repeated

function load_data()
    0
end

function dense_layer(num_input_features, growth_rate, bn_size) # bn_size: bottle_neck_size
    denseLayer = Chain(
        BatchNorm(num_input_features, relu),
        Conv((1, 1), num_input_features => bn_size * growth_rate),
        BatchNorm(bn_size * growth_rate),
        Conv((3, 3), bn_size * growth_rate => growth_rate, pad = (1, 1))
    )
    denseLayer
end

struct denseBlock
    num_input_features
    num_layers
    growth_rate
    bn_size
end

function dense_block_function(block, x)
    outs = []
    catX = x
    growth_rate = block.growth_rate
    bn_size = block.bn_size
    for l in 1:block.num_layers
        num_input_features = size(catX)[3]
        out = dense_layer(num_input_features, growth_rate, bn_size)(catX)
        push!(outs, out)
        catX = cat(catX, out, dims = 3)
    end
    outs[length(outs)]
end

(block::denseBlock)(x) = dense_block_function(block, x)


function dense_net_model()
