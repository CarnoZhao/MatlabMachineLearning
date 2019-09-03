using CUDAdrv, CuArrays
using Flux
using Flux: onehotbatch, crossentropy, onecold, glorot_normal, throttle
using HDF5, Statistics
using Base.Iterators: repeated

function load_data()
    train_data = read(h5open("/data/tongxueqing/zhaox/train_signs.h5"))
    test_data = read(h5open("/data/tongxueqing/zhaox/test_signs.h5"))
    X = Float64.(train_data["train_set_x"]) / 255  |> gpu
    Y = Array(onehotbatch(train_data["train_set_y"], 0:5)) |> gpu
    tX = Float64.(test_data["test_set_x"]) / 255 |> gpu
    tY = Array(onehotbatch(test_data["test_set_y"], 0:5)) |> gpu
    X = permutedims(X, [2, 3, 1, 4])
    tX = permutedims(tX, [2, 3, 1, 4])
    X, Y, tX, tY
end

function model(num_iterations, learning_rate = 0.0001, beta1=  0.9, beta2 = 0.999)
    X, Y, tX, tY = load_data()
    convnet = Chain(
        Conv((5, 5), 3 => 8, pad = (2, 2), relu),
        MaxPool((8, 8)),
        Conv((3, 3), 8 => 16, pad = (1, 1), relu),
        MaxPool((4, 4)),
        x -> reshape(x, :, size(x)[4]),
        Dense(64, 20, relu, initW = glorot_normal),
        Dense(20, 6, initW = glorot_normal),
        softmax
    ) |> gpu
    loss(x, y) = crossentropy(convnet(x), y)
    accuracy(x, y) = mean(onecold(convnet(x)) .== onecold(y))
    parameters = params(convnet)
    data = repeated((X, Y), num_iterations)
    callback = () -> @show(loss(X, Y))
    optimizer = AdaMax(learning_rate, (beta1, beta2))
    Flux.train!(loss, parameters, data, optimizer, cb = throttle(callback, 5))
    println("Accuracy in training: $(round(accuracy(X, Y) * 100, digits = 4))%")
    println("Accuracy in test: $(round(accuracy(tX, tY) * 100, digits = 4))%")
    convnet
end

convnet = model(3000, 0.0005, 0.9, 0.999);
# Accuracy in training: 100.0%
# Accuracy in test: 89.1667%