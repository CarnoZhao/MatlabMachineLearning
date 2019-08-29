using Flux, Statistics
using Flux.Tracker

x = randn(3, 100)
y = ifelse.(sum(x[1:2, :], dims = 1) .< 0, 0, 1)
gn = Flux.glorot_normal
model = Chain(
    Dense(3, 4, relu, initW = gn), 
    Dense(4, 2, relu, initW = gn), 
    Dense(2, 1, sigmoid, initW = gn)
    )
model(x)
ps = params(model)

loss(x, y) = mean(Flux.binarycrossentropy.(model(x), y))
accuracy(x, y) = mean(.!xor.(model(x) .> 0.5, y .== 1))
accuracy(x, y)
ps
data = Iterators.repeated((x, y), 1000)
Flux.train!(loss, ps, data, Flux.Optimise.ADAM())

ps
accuracy(x, y)