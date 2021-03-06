mutable struct KNNAnom
    knndata
    k
    v
    KNNAnom(k::Int, v::Symbol = :gamma) = new(nothing, k, v)
end

ScikitLearn.fit!(m::KNNAnom, x) = ScikitLearn.fit!(m, x, m.v)
function ScikitLearn.fit!(m::KNNAnom, x, v::Symbol)
    m.knndata = kNN.KNNAnomaly(collect(x'), v)
end

ScikitLearn.decision_function(m::KNNAnom, x, k) = .- StatsBase.predict(m.knndata, collect(x'), k)
ScikitLearn.decision_function(m::KNNAnom, x) = ScikitLearn.decision_function(m, x, m.k)
