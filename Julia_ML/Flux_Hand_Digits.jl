using CUDAdrv, CuArrays
using Flux
using Flux: onehotbatch, crossentropy, onecold, glorot_normal, throttle
using HDF5, Statistics
using Base.Iterators: repeated

function load_data()
    train_data = read(h5open("/data/tongxueqing/zhaox/train_signs.h5"))
    test_data = read(h5open("/data/tongxueqing/zhaox/test_signs.h5"))
    X = Float64.(train_data["train_set_x"]) / 255
    Y = Array(onehotbatch(train_data["train_set_y"], 0:5))
    tX = Float64.(test_data["test_set_x"]) / 255
    tY = Array(onehotbatch(test_data["test_set_y"], 0:5))
    X, Y, tX, tY
end

function model(num_iterations = 3000, learning_rate = 0.0001, beta1 = 0.9, beta2 = 0.999)
    X, Y, tX, tY = load_data()
    train_m = size(X)[ndims(X)]
    test_m = size(tX)[ndims(tX)]
    X = reshape(X, :, train_m) |> gpu
    tX = reshape(tX, :, test_m) |> gpu
    Y = reshape(Y, :, train_m) |> gpu
    tY = reshape(tY, :, test_m) |> gpu
    net = Chain(
        Dense(size(X)[1], 100, relu, initW = glorot_normal),
        Dense(100, 50, relu, initW = glorot_normal),
        Dense(50, 20, relu, initW = glorot_normal),
        Dense(20, 10, relu, initW = glorot_normal),
        Dense(10, size(Y)[1], initW = glorot_normal),
        softmax
    ) |> gpu
    loss(x, y) = crossentropy(net(x), y)
    accuracy(x, y) = mean(onecold(net(x)) .== onecold(y))
    parameters = params(net)
    data = repeated((X, Y), 5000)
    callback = () -> @show(loss(X, Y))
    optimizer = AdaMax(learning_rate, (beta1, beta2))
    Flux.train!(loss, parameters, data, optimizer, cb = throttle(callback, 5))
    println("Accuracy in training: $(round(accuracy(X, Y) * 100, digits = 4))%")
    println("Accuracy in test: $(round(accuracy(tX, tY) * 100, digits = 4))%")
    net
end

net = model(3000, 0.0001, 0.9, 0.9999);
# best results
# Accuracy in training: 100.0%
# Accuracy in test: 87.50%