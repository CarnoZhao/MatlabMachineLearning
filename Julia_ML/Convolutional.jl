using HDF5, Statistics
using CUDAdrv, CuArrays, GPUArrays, CUDAnative
GPUArrays.allowscalar(true)
use_gpu = true
dev = use_gpu ? CuArrays.CuArray : Array
sum = use_gpu ? CUDAnative.sum : Base.sum
max = Base.max

onehot(Y, rng) = Int.(rng .== reshape(Y, 1, :))
onecold(Y) = [Int(idx[1]) - 1 for idx in argmax(Y, dims = 1)]
relu(x) = max.(0, x)
drelu(x) = ifelse.(x .> 0, 1, 0)
softmax(X) = exp.(X) ./ sum(exp.(X), dims = 1)
slice(h, w, f, s) = ((h - 1) * s + 1:(h - 1) * s + f, (w - 1) * s + 1:(w - 1) * s + f)
devz(x...) = dev(zeros(x...))
devc(x) = dev(zeros(size(x)))
crossentropy(AL, Y) = -mean(sum(Y .* log.(AL), dims = 1))
dims_mapping(nH, nW, f, p, s) = (Int((nH + 2 * p - f) / s + 1), Int((nW + 2 * p - f) / s + 1))
flat(P) = (reshape(P, :, size(P)[ndims(P)]), size(P))

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

function init_cvParams(X_dims, cvDims)
    cvParams = []
    nH, nW, nCin, m = X_dims
    for cvdim in cvDims
        f, s, p, nCout = cvdim
        W = randn(f, f, nCin, nCout) * sqrt(2 / (nH * nW * nCin))
        b = zeros(1, 1, nCout)
        push!(cvParams, [dev(W), dev(b)])
        nH, nW = dims_mapping(nH, nW, f, p, s)
        nCin = nCout
    end
    cvV = [[devc(para) for para in paras] for paras in cvParams]
    cvS = deepcopy(cvV)
    cvParams, cvS, cvV
end

function init_fcParams(fcDims)
    fcParams = []
    L = length(fcDims) - 1
    for i in 1:L
        W = randn(fcDims[i + 1], fcDims[i]) * sqrt(2 / fcDims[i])
        b = zeros(fcDims[i + 1], 1)
        push!(fcParams, [dev(W), dev(b)])
    end
    fcV = [[devc(para) for para in paras] for paras in fcParams]
    fcS = deepcopy(fcV)
    fcParams, fcV, fcS
end

function padding(X, p)
    nH, nW, nC, m = size(X)
    newX = devz(nH + 2 * p, nW + 2 * p, nC, m)
    newX[p + 1:p + nH, p + 1:p + nW, :, :] = X
    newX
end

function conv(X, params, hparams)
    W, b = params
    s, p = hparams
    nHin, nWin, nCin, m = size(X)
    f, f, nCin, nCout = size(W)
    X = padding(X, p)
    nH, nW = dims_mapping(nHin, nWin, f, p, s)
    Z = devz(nH, nW, nCout, m)
    for h in 1:nH, w in 1:nW
        hSlice, wSlice = slice(h, w, f, s)
        tmp = reshape(X[hSlice, wSlice, :, :], f, f, nCin, 1, m) .* reshape(W, f, f, nCin, nCout, 1)
        Z[h, w, :, :] = reshape(sum(tmp, dims = 1:3), 1, 1, nCout, m)
    end
    relu(Z .+ b)
end 

function pool(Z, hparams)
    Z = Array(Z)
    f, s, mode = hparams
    nHin, nWin, nCin, m = size(Z)
    nH, nW = dims_mapping(nHin, nWin, f, 0, s)
    P = devz(nH, nW, nCin, m)
    func = mode == "max" ? maximum : mean
    for h in 1:nH, w in 1:nW
        hSlice, wSlice = slice(h, w, f, s)
        P[h, w, :, :] = func(Z[hSlice, wSlice, :, :], dims = 1:2)
    end
    P
end

function fc(A, params, useSoftmax)
    W, b = params
    Z = W * A .+ b
    A = (useSoftmax ? softmax : relu)(Z)
    A
end

function forward(P, cvParams, fcParams, cvDims, plDims, fcDims)
    cvCaches = []
    for cvIdx in 1:length(cvDims)
        A = conv(P, cvParams[cvIdx], cvDims[cvIdx][2:3])
        push!(cvCaches, [P, A])
        P = pool(A, plDims[cvIdx])
    end
    A, flatDims = flat(P)
    fcCaches = [A]
    for fcIdx in 1:length(fcDims) - 1
        A = fc(A, fcParams[fcIdx], fcIdx == length(fcDims) - 1)
        push!(fcCaches, A)
    end
    A, cvCaches, fcCaches, flatDims
end

function fcback(dA, fcCaches, fcParams, fcIdx, m)
    A, A_prev = fcCaches[fcIdx + 1], fcCaches[fcIdx]
    W = fcParams[fcIdx][1]
    dZ = dA .* ifelse(fcIdx == length(fcCaches) - 1, 1, drelu(A))
    dW = dZ * A_prev' / m
    db = mean(dZ, dims = 2)
    dA = W' * dZ
    dA, dW, db
end

function poolback(dP, A, cvIdx, plDims, m)
    nH, nW, nCout = size(dP)
    plf, pls, mode = plDims[cvIdx]
    dA = devc(A)
    dreluA = drelu(A)
    dP = Array(dP)
    A = Array(A)
    for h in 1:nH, w in 1:nW
        hSlice, wSlice = slice(h, w, plf, pls)
        if mode == "max"
            Aslice = A[hSlice, wSlice, :, :]
            dpool = Aslice .== maximum(Aslice, dims = 1:2)
        else
            dpool = ones(plf, plf, nCout, m) / plf ^ 2
        end
        dA[hSlice, wSlice, :, :] .+= dev(dpool .* dP[h:h, w:w, :, :])
    end
    dA .* dreluA
end

function convback(dZ, Pprev, cvParams, cvDims, cvIdx, m)
    W, b = cvParams[cvIdx]
    f, s, p, nCout = cvDims[cvIdx]
    f, f, nCin, nCout = size(W)
    nH, nW, nCout, m = size(dZ)
    dP = devz(size(Pprev) .+ (2 * p, 2 * p, 0, 0))
    dW, db = devc(W), devc(b)
    Pprev = padding(Pprev, p)
    for h in 1:nH, w in 1:nW
        hSlice, wSlice = slice(h, w, f, s)
        tmp = reshape(W, f, f, nCin, nCout, 1) .* reshape(dZ[h, w, :, :], 1, 1, 1, nCout, m)
        dP[hSlice, wSlice, :, :] .+= dropdims(sum(tmp, dims = 4), dims = 4)
        tmp = reshape(Pprev[hSlice, wSlice, :, :], f, f, nCin, 1, m) .* reshape(dZ[h, w, :, :], 1, 1, 1, nCout, m)
        dW .+= dropdims(mean(tmp, dims = 5), dims = 5)
    end
    db = dropdims(mean(sum(dZ, dims = 1:2), dims = 4), dims = 4)
    dP = dP[p + 1:size(dP)[1] - p, p + 1:size(dP)[2] - p, :, :]
    dP, dW, db
end

function backward(AL, Y, cvCaches, fcCaches, cvParams, fcParams, cvDims, plDims, flatDims)
    m = size(Y)[2]
    dA = AL - Y
    fcGrads = []
    for fcIdx in length(fcCaches) - 1:-1:1
        dA, dW, db = fcback(dA, fcCaches, fcParams, fcIdx, m)
        push!(fcGrads, [dW, db])
    end
    dP = reshape(dA', flatDims)
    cvGrads = []
    for cvIdx in length(cvCaches):-1:1
        Pprev, A = cvCaches[cvIdx]
        dZ = poolback(dP, A, cvIdx, plDims, m)
        dP, dW, db = convback(dZ, Pprev, cvParams, cvDims, cvIdx, m)
        push!(cvGrads, [dW, db])
    end
    cvGrads, fcGrads
end

function update_parameters(cvParams, fcParams, cvGrads, fcGrads, learnRate, beta1, beta2, epsilon, cvV, cvS, fcV, fcS, t)
    combines = ((cvParams, cvGrads, cvV, cvS), (fcParams, fcGrads, fcV, fcS))
    for (P, G, V, S) in combines, i in 1:length(P), j in (1, 2)
        L = length(P)
        V[i][j] = beta1 * V[i][j] + (1 - beta1) * G[L - i + 1][j]
        S[i][j] = beta2 * S[i][j] + (1 - beta2) * G[L - i + 1][j] .^ 2
        VCorr = V[i][j] / (1 - beta1 ^ t)
        SCorr = S[i][j] / (1 - beta2 ^ t)
        P[i][j] -= learnRate * VCorr ./ (sqrt.(SCorr) .+ epsilon)
    end
end

function predict(X, Y, cvParams, fcParams, cvDims, plDims, fcDims, name)
    AL, _, _, _ = forward(X, cvParams, fcParams, cvDims, plDims, fcDims)
    accuarcy = mean(onecold(AL) .== onecold(Y)) * 100
    println("Accuracy in $(name) set: $(accuarcy)")
end

function cnn(X, Y, cvDims, plDims, fcDims, num_iterations, learnRate, beta1, beta2, epsilon)
    cvParams, cvV, cvS = init_cvParams(size(X), cvDims)
    fcParams, fcV, fcS = init_fcParams(fcDims)
    for t in 1:num_iterations
        AL, cvCaches, fcCaches, flatDims = forward(X, cvParams, fcParams, cvDims, plDims, fcDims)
        cvGrads, fcGrads = backward(AL, Y, cvCaches, fcCaches, cvParams, fcParams, cvDims, plDims, flatDims)
        update_parameters(cvParams, fcParams, cvGrads, fcGrads, learnRate, beta1, beta2, epsilon, cvV, cvS, fcV, fcS, t)
        if t % 1 == 0
            cost = crossentropy(AL, Y)
            println("Cost after iteration $(t): $(cost)")
        end
    end
    cvParams, fcParams
end

function main(num_iterations = 100, learnRate = 0.0001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
    X, Y, tX, tY = load_data()
    cvDims = [[5, 1, 2, 5], [3, 1, 1, 10]]
    plDims = [[8, 8, "max"], [4, 4, "max"]]
    fcDims = [40, 20, 6]
    cvParams, fcParams = cnn(X, Y, cvDims, plDims, fcDims, num_iterations, learnRate, beta1, beta2, epsilon)
    predict(X, Y, cvParams, fcParams, cvDims, plDims, fcDims, "train")
    predict(tX, tY, cvParams, fcParams, cvDims, plDims, fcDims, "test")
    cvParams, fcParams
end

cvParams, fcParams = main();