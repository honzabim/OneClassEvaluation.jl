module OneClassEvaluation

    using kNN
    using ADatasets
    using StatsBase
    using ScikitLearn

    include("knn.jl")
    include("evaluation.jl")
end
