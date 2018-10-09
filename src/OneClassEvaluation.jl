module OneClassEvaluation

    using kNN
    using ADatasets
    using StatsBase
    using ScikitLearn
    using DataFrames
    using CSV

    include("knn.jl")
    include("evaluation.jl")
end
