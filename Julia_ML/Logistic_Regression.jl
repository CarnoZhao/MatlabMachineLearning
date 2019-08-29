using PyPlot;
using LinearAlgebra, Statistics, HDF5, Images

train_data = read(h5open("/pnas/liujiang_group/yinyao/codes/repos/DataLists/train_catvnoncat.h5"))
test_data = read(h5open("/pnas/liujiang_group/yinyao/codes/repos/DataLists/test_catvnoncat.h5"))

train_X = Int.(train_data["train_set_x"]) ./ 255
train_Y = train_data["train_set_y"]
test_X = Int.(test_data["test_set_x"]) ./ 255
test_Y = test_data["test_set_y"]

function sigmoid(x)
    1 / (1 + exp(-x))
end

function zero_init(dim)
    w = zeros(dim)
    b = 0
    w, b
end

function propagate(w, b, X, Y)
    m = size(X)[2]
    Z = w' * X .+ b
    A = sigmoid.(Z)
    cost = -mean(Y .* log.(A) .+ (1 .- Y) .* log.(1 .- A))
    dw = X * (A - Y)' ./ m
    db = mean(A - Y)
    dw, db, cost
end

function optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = false)
    for i in 1:num_iterations
        dw, db, cost = propagate(w, b, X, Y)
        w -= learning_rate .* dw
        b -= learning_rate * db
        if print_cost && i % 100 == 0
            println("Cost after iteration $(i): $(round(cost, digits = 4))")
        end
    end
    w, b
end

function predict(w, b, X)
    m = size(X)[2]
    A = sigmoid.(w' * X .+ b)
    Y_pred = ifelse.(A .<= 0.5, 0, 1)
end

function model(train_X, train_Y, test_X, test_Y; num_iterations = 2000, learning_rate = 0.009, print_cost = false)
    train_X = reshape(train_X, :, size(train_X)[length(size(train_X))])
    train_Y = reshape(train_Y, :, size(train_Y)[length(size(train_Y))])
    test_X = reshape(test_X, :, size(test_X)[length(size(test_X))])
    test_Y = reshape(test_Y, :, size(test_Y)[length(size(test_Y))])
    n = size(train_X)[1]    
    w, b = zero_init(n)
    w, b = optimize(w, b, train_X, train_Y, num_iterations, learning_rate, print_cost)
    train_Y_pred = predict(w, b, train_X)
    test_Y_pred = predict(w, b, test_X)
    
    println("train accuracy: $(round(mean(train_Y_pred .== train_Y) * 100, digits = 2))%")
    println("test accuracy: $(round(mean(test_Y_pred .== test_Y) * 100, digits = 2))%")
    w, b
end

w, b = model(train_X, train_Y, test_X, test_Y, num_iterations = 2000, print_cost = true);

newimg = load("/pnas/liujiang_group/yinyao/codes/repos/DataLists/cat.jfif")
newimg = channelview(newimg)
newimg = imresize(newimg, 3, 64, 64)
X = reshape(newimg, :, 1)
for i in predict(w, b, X)
    if i == 1
        println("Prediction $(i): Cat")
    else
        println("Prediction $(i): Non-cat")
    end
end