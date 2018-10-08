mutable struct KNNAnom
    knndata
    k
    v
    KNNAnom(k::Int, v::Symbol = :gamma) = new(nothing, k, v)
end

fit!(m::KNNAnom, x) = fit!(m, x, m.v)
function fit!(m::KNNAnom, x, v::Symbol)
    m.knndata = KNNAnomaly(x, v)
end

predict(m::KNNAnom, x, k) = StatsBase.predict(m, x, k)
predict(m::KNNAnom, x) = StatsBase.predict(m, x, m.k)

decision_function(m::KNNAnom, x) = predict(m, x)
