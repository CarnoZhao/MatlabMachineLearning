using HDF5, Statistics
using CUDAdrv, CuArrays

onehot(Y, rng) = Int.(rng .== reshape(Y, 1, :))
relu(x) = max(0, x)
drelu(x) = ifelse(x > 0, 1, 0)
softmax(X) = exp.(X) / sum(exp.(X))

function load_data()
    train_data = read(h5open("/data/tongxueqing/zhaox/train_signs.h5"))
    test_data = read(h5open("/data/tongxueqing/zhaox/test_signs.h5"))
    X = Float64.(train_data["train_set_x"]) / 255
    Y = onehot(train_data["train_set_y"], 0:5)
    tX = Float64.(test_data["test_set_x"]) / 255
    tY = onehot(test_data["test_set_y"], 0:5)
    X, Y, tX, tY
end

function init_conv_parameters(X_dims, conv_dims)
    conv_parameters = []
    nH, nW, nc_in, m = X_dims
    for conv_dim in conv_dims
        f, s, p, nc_out = conv_dim
        W = randn(f, f, nc_in, nc_out) * sqrt(2 / (nH * nW * nc_in))
        b = zeros(1, 1, 1, nc_out)
        push!(conv_parameters, [W, b])
        nH = Int((nH + 2 * p - f) / s + 1)
        nW = Int((nW + 2 * p - f) / s + 1)
        nc_in = nc_out
    end
    conv_parameters
end

function init_fc_parameters(fc_dims)
    fc_parameters = []
    L = length(fc_dims) - 1
    for i in 1:L
        W = randn(fc_dims[i], fc_dims[i + 1]) * sqrt(2 / fc_dims[i])
        b = zeros(fc_dims[i + 1])
        push!(fc_parameters, [W, b])
    end
    fc_parameters
end

function padding(X, p)
    nH, nW, nc, m = size(X)
    newX = zeros(nH + 2 * p, nW + 2 * p, nc)
    newX[p + 1:p + nH, p + 1:p + nW, :, :] = X
    newX
end

function conv_forward(X, W, b, f, s, p, nc_out)
    nH_in, nW_in, nc_in, m = size(X)
    X = padding(X, p)
    nH = Int((nH_in + 2 * p - f) / s + 1)
    nW = Int((nW_in + 2 * p - f) / s + 1)
    Z = zeros(nH, nW, nc_out, m)
    for h in 1:nH, w in 1:nW, i in 1:m
        x = X[h * s + 1:h * s + f, w * s + 1: w * s + f, :, i]
        Z[h, w, :, i] = sum(x .* W, dims = (1, 2, 4)) .+ b
    end
    Z
end 

function pool_forward(Z, f, s, mode)
    nH_in, nW_in, nc_in, m = size(Z)
    nH = Int((nH_in - f) / s + 1)
    nW = Int((nW_in - f) / s + 1)
    P = zeros(nH, nW, nc_in, m)
    for h in 1:nH, w in 1:nW, i in 1:m
        P[h, w, :, i] = (mode == "max" ? maximum: mean)(Z[h * s + 1:h * s + f, w * s + 1:w * s + f, :, i], dims = 1:2)
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
        dA = zeros(size(A) .+ 2 * p)
        dW = zeros(size(W))
        db = zeros(b)
        for h in 1:nH, w in 1:nW, i in 1:m
            h_slice = h * s_pool + 1:h * s_pool + f_pool
            w_slice = w * s_pool + 1:w * s_pool + f_pool
            if mode == "max"
                A_slice = A[h_slice, w_slice, :, i]
                dpool = A_slice .== maximum(A_slice, dims = (1, 2, 4))
            else
                dpool = ones(f_pool, f_pool) / f_pool ^ 2
            dA[h_slice, w_slice, :, i] .+= dpool .* dP[h, w, :, i]
        end
        dZ = dA .* drelu(A)
        nH, nW, nc = size(dZ)
        for h in 1:nH, w in 1:nW, i in 1:m
            h_slice = h * s + 1:h * s + f
            w_slice = w * s + 1:w * s + f
            dP[h_slice, w_slice, :, i] .+= W .* dZ[h, w, :, i]
            dW .+= P_prev[h_slice, w_slice, :, i] .* dZ[h, w, :, i] / m
            db .+= dZ[h, w, :, i] / m
            push!(conv_grads, [dW, db])
        end
    end
    conv_grads, fc_grads
end

function update_parameters(conv_parameters, fc_parameters, conv_grads, fc_grads, learning_rate, beta1, beta2, epsilon, conv_V, conv_S, fc_V, fc_S)
    
end


