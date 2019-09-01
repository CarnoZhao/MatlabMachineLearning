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
    m, nH, nW, nc_in = X_dims
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
    m, nH, nW, nc = size(X)
    newX = zeros(nH + 2 * p, nW + 2 * p, nc)
    newX[:, p + 1:p + nH, p + 1:p + nW, :] = X
    newX
end

function conv_forward(X, W, b, f, s, p, nc_out)
    # X = m * nH_in * nW_in * nc_in
    # W = f * f * nc_out
    # b = 1 * 1 * 1 * nc_out
    # Z = m *  nH * nW * nc_out
    m, nH_in, nW_in, nc_in = size(X)
    X = padding(X, p)
    nH = Int((nH_in + 2 * p - f) / s + 1)
    nW = Int((nW_in + 2 * p - f) / s + 1)
    Z = zeros(m, nH, nW, nc_out)
    for h in 1:nH for w in 1:nW for i in 1:m
        x = X[i, h * s + 1:h * s + f, w * s + 1: w * s + f, :]
        Z[i, h, w, :] = sum(x .* W, dims = 1:3) .+ b
    end end end
    Z
end 

function pool_forward(Z, f, s, mode)
    m, nH_in, nW_in, nc_in = size(Z)
    nH = Int((nH_in - f) / s + 1)
    nW = Int((nW_in - f) / s + 1)
    P = zeros(m, nH, nW, nc_in)
    for h in 1:nH for w in 1:nW for i in 1:m
        P[i, h, w, :] = max(Z[i, h * s + 1:h * s + f, w * s + 1:w * s + f, :], dims = 1:2)
    end end end
    P
end

function forward(X, conv_parameters, fc_parameters, conv_dims, pool_dims, fc_dims)
    conv_caches = []
    fc_caches = []
    A = X
    for conv_idx in 1:length(conv_dims)
        W, b = conv_parameters[conv_idx]
        f, s, p, nc_out = conv_dims[conv_idx]
        f_pool, s_pool, mode = pool_dims[conv_idx]
        Z = conv_forward(A, W, b, f, s, p, nc_out)
        Z = relu.(Z)
        P = pool_forward(Z, f_pool, s_pool, mode)
        A = P
    end
    flat_dims = size(A)
    A = reshape(A, size(A)[1], :)'
    for fc_idx in 1:length(fc_dims)
        W, b = fc_parameters[fc_idx]
        Z = W * A .+ b
        A = (fc_idx == length(fc_dims) ? softmax : relu)(Z)
    end
    A, conv_caches, fc_caches, flat_dims
end

function backward(A, Y, conv_caches, fc_caches, conv_parameters, fc_parameters)
    m = size(Y)[2]
    dA = A - Y
    conv_grads = []
    fc_grads = []
    for fc_idx in length(fc_caches):-1:1
        A, A_prev, W = fc_caches[fc_idx]
        dZ = dA .* ifelse(fc_idx == length(fc_caches), 1, drelu(A_prev))
        dW = dZ * A_prev' / m
        db = mean(dZ, dims = 1)
        dA = W' * dZ
    end
    dP = reshape(dA', )
