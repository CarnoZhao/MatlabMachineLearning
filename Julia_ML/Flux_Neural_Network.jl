using CUDAdrv
using CuArrays
using Flux
using Flux: onehotbatch, crossentropy, onecold, glorot_normal, throttle
using MLDatasets, Statistics, LinearAlgebra
using Base.Iterators: repeated

function load_data()
    train_x, train_y = MLDatasets.MNIST.traindata(Float64)
    test_x, test_y = MLDatasets.MNIST.testdata(Float64)
    train_m = size(train_x)[length(size(train_x))]
    test_m = size(test_x)[length(size(test_x))]
    X = reshape(train_x, :, train_m) |> gpu
    Y = Array(onehotbatch(train_y, 0:9)) |> gpu
    tX = reshape(test_x, :, test_m) |> gpu
    tY = Array(onehotbatch(test_y, 0:9)) |> gpu
    X, Y, tX, tY
end

function model()
    X, Y, tX, tY = load_data()
    chain = Chain(
        Dense(size(X)[1], 20, relu, initW = glorot_normal),
        Dense(20, 15, relu, initW = glorot_normal),
        Dense(15, size(Y)[1], initW = glorot_normal),
        softmax
    ) |> gpu
    loss(x, y) = crossentropy(chain(x), y)
    accuracy(x, y) = mean(onecold(chain(x)) .== onecold(y))
    ps = params(chain)
    data = repeated((X, Y), 1500)
    evalcb = () -> @show(loss(X, Y))
    opt = AdaMax()
    Flux.train!(loss, ps, data, opt, cb = throttle(evalcb, 1))
    println("Accuracy in training: $(accuracy(X, Y) * 100)%")
    println("Accuracy in test: $(accuracy(tX, tY) * 100)%")
end

model();
# Accuracy in training: 96.25333333333333%
# Accuracy in test: 95.22%
