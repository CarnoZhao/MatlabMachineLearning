using HDF5, Statistics
using CUDAdrv, CuArrays
use_gpu = true
dev = use_gpu ? CuArrays.CuArray : Array

onehot(Y, rng) = Int.(rng .== reshape(Y, 1, :))
onecold(Y) = [Int(idx[1]) for idx in argmax(Y, dims = 1)]
relu(x) = max(0, x)
drelu(x) = ifelse(x > 0, 1, 0)
softmax(X) = exp.(X) / sum(exp.(X))
slice(h, w, f, s) = ((h - 1) * s + 1:(h - 1) * s + f, (w - 1) * s + 1:(w - 1) * s + f)

function load_data()
    train_data = read(h5open("/data/tongxueqing/zhaox/train_signs.h5"))
    test_data = read(h5open("/data/tongxueqing/zhaox/test_signs.h5"))
    X = Float64.(train_data["train_set_x"]) / 255
    Y = onehot(train_data["train_set_y"], 0:5)
    tX = Float64.(test_data["test_set_x"]) / 255
    tY = onehot(test_data["test_set_y"], 0:5)
    X = permutedims(X, [2, 3, 1, 4])
    tX = permutedims(tX, [2, 3, 1, 4])
    [dev(arr) for arr in (X, Y, tX, tY)]
end

function init_conv_parameters(X_dims, conv_dims)
    conv_parameters = []
    nH, nW, nc_in, m = X_dims
    for conv_dim in conv_dims
        f, s, p, nc_out = conv_dim
        W = randn(f, f, nc_in, nc_out) * sqrt(2 / (nH * nW * nc_in))
        b = zeros(1, 1, 1, nc_out)
        push!(conv_parameters, [dev(W), dev(b)])
        nH = Int((nH + 2 * p - f) / s + 1)
        nW = Int((nW + 2 * p - f) / s + 1)
        nc_in = nc_out
    end
    conv_V = [[dev(zeros(size(para))) for para in paras] for paras in conv_parameters]
    conv_S = deepcopy(conv_V)
    conv_parameters, conv_S, conv_V
end

function init_fc_parameters(fc_dims)
    fc_parameters = []
    L = length(fc_dims) - 1
    for i in 1:L
        W = randn(fc_dims[i], fc_dims[i + 1]) * sqrt(2 / fc_dims[i])
        b = zeros(fc_dims[i + 1])
        push!(fc_parameters, [dev(W), dev(b)])
    end
    fc_V = [[dev(zeros(size(para))) for para in paras] for paras in fc_parameters]
    fc_S = deepcopy(fc_V)
    fc_parameters, fc_V, fc_S
end

function padding(X, p)
    nH, nW, nc, m = size(X)
    newX = dev(zeros(nH + 2 * p, nW + 2 * p, nc, m))
    newX[p + 1:p + nH, p + 1:p + nW, :, :] = X
    newX
end

function conv_forward(X, W, b, f, s, p, nc_out)
    nH_in, nW_in, nc_in, m = size(X)
    X = padding(X, p)
    nH = Int((nH_in + 2 * p - f) / s + 1)
    nW = Int((nW_in + 2 * p - f) / s + 1)
    Z = dev(zeros(nH, nW, nc_out, m))
    for h in 1:nH, w in 1:nW, i in 1:m
        h_slice, w_slice = slice(h, w, f, s)
        Z[h, w, :, i] = sum(X[h_slice, w_slice, :, i] .* W, dims = (1, 2, 4)) .+ b
        println("$(h), $(w), $(i)")
    end
    Z
end 

function pool_forward(Z, f, s, mode)
    nH_in, nW_in, nc_in, m = size(Z)
    nH = Int((nH_in - f) / s + 1)
    nW = Int((nW_in - f) / s + 1)
    P = dev(zeros(nH, nW, nc_in, m))
    for h in 1:nH, w in 1:nW, i in 1:m
        h_slice, w_slice = slice(h, w, f, s)
        P[h, w, :, i] = (mode == "max" ? maximum : mean)(Z[h_slice, w_slice, :, i], dims = 1:2)
    end
    P
end

function forward(X, conv_parameters, fc_parameters, conv_dims, pool_dims, fc_dims)
    A = X
    conv_caches = []
    for conv_idx in 1:length(conv_dims)
        push!(conv_caches, [A])
        W, b = conv_parameters[conv_idx]
        f, s, p, nc_out = conv_dims[conv_idx]
        f_pool, s_pool, mode = pool_dims[conv_idx]
        Z = conv_forward(A, W, b, f, s, p, nc_out)
        Z = relu.(Z)
        push!(conv_caches[length(conv_caches)], Z)
        P = pool_forward(Z, f_pool, s_pool, mode)
        A = P
    end
    flat_dims = size(A)
    A = reshape(A, :, flat_dims[ndims(A)])
    fc_caches = [A]
    for fc_idx in 1:length(fc_dims)
        W, b = fc_parameters[fc_idx]
        Z = W * A .+ b
        A = (fc_idx == length(fc_dims) ? softmax : relu)(Z)
        push!(fc_caches, A)
    end
    A, conv_caches, fc_caches, flat_dims
end

function backward(AL, Y, conv_caches, fc_caches, conv_parameters, fc_parameters, conv_dims, pool_dims, flat_dims)
    m = size(Y)[2]
    dA = AL - Y
    fc_grads = []
    for fc_idx in length(fc_caches):-1:2
        A, A_prev = fc_caches[fc_idx], fc_caches[fc_idx - 1]
        W = fc_parameters[fc_idx][1]
        dZ = dA .* ifelse(fc_idx == length(fc_caches), 1, drelu(A))
        dW = dZ * A_prev' / m
        db = mean(dZ, dims = 1)
        dA = W' * dZ
        push!(fc_grads, [dW, db])
    end
    dP = reshape(dA', flat_dims)
    conv_grads = []
    for conv_idx in length(conv_caches):-1:1
        nH, nW, nc = size(dP)
        P_prev, A = conv_caches[conv_idx]
        W, b = conv_parameters[conv_idx]
        f_pool, s_pool, mode = pool_dims[conv_idx]
        f, s, p, nc = conv_dims[conv_idx]
        dA = zeros(size(A) .+ [2 * p, 2 * p, 0, 0])
        dW = zeros(size(W))
        db = zeros(b)
        for h in 1:nH, w in 1:nW, i in 1:m
            h_slice, w_slice = slice(h, w, f_pool, s_pool)
            if mode == "max"
                A_slice = A[h_slice, w_slice, :, i]
                dpool = A_slice .== maximum(A_slice, dims = (1, 2, 4))
            else
                dpool = ones(f_pool, f_pool) / f_pool ^ 2
                dA[h_slice, w_slice, :, i] .+= dpool .* dP[h, w, :, i]
            end
            dA = dA[p + 1:size(dA)[1] - p, p + 1:size(dA)[2] - p, :, :]
        end
        dZ = dA .* drelu(A)
        nH, nW, nc = size(dZ)
        for h in 1:nH, w in 1:nW, i in 1:m
            h_slice, w_slice = slice(h, w, f, s)
            dP[h_slice, w_slice, :, i] .+= W .* dZ[h, w, :, i]
            dW .+= P_prev[h_slice, w_slice, :, i] .* dZ[h, w, :, i] / m
            db .+= dZ[h, w, :, i] / m
            push!(conv_grads, [dW, db])
        end
    end
    conv_grads, fc_grads
end

function update_parameters(conv_parameters, fc_parameters, conv_grads, fc_grads, learning_rate, beta1, beta2, epsilon, conv_V, conv_S, fc_V, fc_S, t)
    combines = ((conv_parameters, conv_grads, conv_V, conv_S), (fc_parameters, fc_grads, fc_V, fc_S))
    for (P, G, V, S) in combines, i in 1:length(P), j in (1, 2)
        L = length(P)
        V[i][j] = beta1 * V[i][j] + (1 - beta1) * G[L - i + 1][j]
        S[i][j] = beta2 * S[i][j] + (1 - beta2) * G[L - i + 1][j] .^ 2
        V_cor = V[i][j] / (1 - beta1 ^ t)
        S_cor = S[i][j] / (1 - beta2 ^ t)
        P[i][j] -= learning_rate * V_cor ./ (sqrt.(S_cor) .+ epsilon)
    end
end

function compute_cost(AL, Y)
    mean(sum(Y .* log.(AL), dims = 1))
end

function predict(X, Y, conv_parameters, fc_parameters, conv_dims, pool_dims, fc_dims, name)
    AL, _, _, _ = forward(X, conv_parameters, fc_parameters, conv_dims, pool_dims, fc_dims)
    accuarcy = mean(onecold(AL) .== onecold(Y)) * 100
    println("Accuracy in $(name) set: $(accuarcy)")
end

function cnn(X, Y, conv_dims, pool_dims, fc_dims, num_iterations, learning_rate, beta1, beta2, epsilon)
    conv_parameters, conv_V, conv_S = init_conv_parameters(size(X), conv_dims)
    fc_parameters, fc_V, fc_S = init_fc_parameters(fc_dims)
    for t in 1:num_iterations
        AL, conv_caches, fc_caches, flat_dims = forward(X, conv_parameters, fc_parameters, conv_dims, pool_dims, fc_dims)
        conv_grads, fc_grads = backward(AL, Y, conv_caches, fc_caches, conv_parameters, fc_parameters, conv_dims, pool_dims, flat_dims)
        update_parameters(conv_parameters, fc_parameters, conv_grads, fc_grads, learning_rate, beta1, beta2, epsilon, conv_V, conv_S, fc_V, fc_S, t)
        if t % 1 == 0
            cost = compute_cost(AL, Y)
            println("Cost after iteration $(t): $(cost)")
        end
    end
    conv_parameters, fc_parameters
end

function main(num_iterations = 2, learning_rate = 0.0001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
    X, Y, tX, tY = load_data()
    conv_dims = [[5, 1, 2, 8], [3, 1, 1, 16]]
    pool_dims = [[8, 8, "max"], [4, 4, "max"]]
    fc_dims = [64, 20, 6]
    conv_parameters, fc_parameters = cnn(X, Y, conv_dims, pool_dims, fc_dims, num_iterations, learning_rate, beta1, beta2, epsilon)
    predict(X, Y, conv_parameters, fc_parameters, conv_dims, pool_dims, fc_dims, "train")
    predict(tX, tY, conv_parameters, fc_parameters, conv_dims, pool_dims, fc_dims, "test")
    conv_parameters, fc_parameters
end

conv_parameters, fc_parameters = main()