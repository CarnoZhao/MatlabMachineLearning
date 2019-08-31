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

function model()
    X, Y, tX, tY = load_data()
    train_m = size(X)[ndims(X)]
    test_m = size(tX)[ndims(tX)]
    X = reshape(X, :, train_m) |> gpu
    tX = reshape(tX, :, test_m) |> gpu
    Y = reshape(Y, :, train_m) |> gpu
    tY = reshape(tY, :, test_m) |> gpu
    net = Chain(
        Dense(size(X)[1], 50, relu, initW = glorot_normal),
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
    optimizer = AdaMax()
    Flux.train!(loss, parameters, data, optimizer, cb = throttle(callback, 5))
    println("Accuracy in training: $(accuracy(X, Y) * 100)%")
    println("Accuracy in test: $(accuracy(tX, tY) * 100)%")
    net
end

net = model();
# Accuracy in training: 99.9074074074074%
# Accuracy in test: 80.83333333333333%