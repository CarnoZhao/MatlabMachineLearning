using PyPlot;
using LinearAlgebra, Statistics, HDF5, Images

train_data = read(h5open("D:\\Download\\train_catvnoncat.h5"))
test_data = read(h5open("D:\\Download\\test_catvnoncat.h5"))

train_X = Int.(train_data["train_set_x"]) ./ 255
train_Y = train_data["train_set_y"]
test_X = Int.(test_data["test_set_x"]) ./ 255
test_Y = test_data["test_set_y"]

function initialize_parameters(layer_dims)
    parameters = []
    L = length(layer_dims) - 1
    for i in 1:L
        W = randn(layer_dims[i + 1], layer_dims[i]) * 0.01
        b = zeros(layer_dims[i + 1], 1)
        push!(parameters, [W, b])
    end
    parameters
end

function L_model_forward(X, parameters)
    caches = [X]
    A = X
    L = length(parameters)
    for i in 1:L
        A_prev = A
        W, b = parameters[i]
        Z = W * A_prev .+ b
        A = ifelse(i == L, 1 ./ (1 .+ exp.(-Z)), max.(0, Z))
        push!(caches, A)
    end
    caches
end

function compute_cost(AL, Y)
    cost = -mean(Y .* log.(AL) + (1 .- Y) .* log.(1 .- AL))
    cost
end

function L_model_backward(Y, parameters, caches)
    grads = []
    m = size(Y)[2]
    L = length(parameters)
    AL = caches[length(caches)]
    dA = (AL - Y) ./ (AL .* (1 .- AL))
    for i in L:-1:1
        W = parameters[i][1]
        dZ = ifelse(i == L, 
            dA .* caches[i + 1] .* (1 .- caches[i + 1]),
            ifelse.(caches[i + 1] .> 0, dA, 0))
        dW = dZ * caches[i]' ./ m
        db = mean(dZ, dims = 2)
        dA = W' * dZ
        push!(grads, [dW, db])
    end
    grads
end

function update_parameters(parameters, grads, learning_rate)
    L = length(parameters)
    for i in 1:L
        parameters[i][1] -= learning_rate .* grads[L - i + 1][1]
        parameters[i][2] -= learning_rate .* grads[L - i + 1][2]
    end
    parameters
end

function L_layer_model(X, Y, layer_dims; learning_rate = 0.0075, num_iterations = 3000)
    parameters = initialize_parameters(layer_dims)
    for i in 1:num_iterations
        caches = L_model_forward(X, parameters)
        grads = L_model_backward(Y, parameters, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if i % 100 == 0
            cost = compute_cost(caches[length(caches)], Y)
            println("Cost after iteration $(i): $(round(cost, digits = 4))")
        end
    end
    parameters
end

function predict(X, parameters)
    caches = L_model_forward(X, parameters)
    Y_hat = ifelse.(caches[length(caches)] .> 0.5, 1, 0)
    Y_hat
end

function main(train_X, train_Y, test_X, test_Y; num_iterations = 2500, learning_rate = 0.0075)
    train_X = reshape(train_X, :, size(train_X)[length(size(train_X))])
    train_Y = reshape(train_Y, :, size(train_Y)[length(size(train_Y))])
    test_X = reshape(test_X, :, size(test_X)[length(size(test_X))])
    test_Y = reshape(test_Y, :, size(test_Y)[length(size(test_Y))])
    layer_dims = [size(train_X)[1], 20, 7, 5, size(train_Y)[1]]
    parameters = L_layer_model(train_X, train_Y, layer_dims, num_iterations = num_iterations, learning_rate = learning_rate)
    train_Y_pred = predict(train_X, parameters)
    test_Y_pred = predict(test_X, parameters)
    println("train accuracy: $(round(mean(train_Y_pred .== train_Y) * 100, digits = 2))%")
    println("test accuracy: $(round(mean(test_Y_pred .== test_Y) * 100, digits = 2))%")
    parameters
end

parameters = main(train_X, train_Y, test_X, test_Y);