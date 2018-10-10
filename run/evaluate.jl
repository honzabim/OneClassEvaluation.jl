# folderpath = "D:/dev/julia/"
folderpath = "/home/bimjan/dev/julia/"
const dataPath = folderpath * "data/loda/public/datasets/numerical"
push!(LOAD_PATH, folderpath, folderpath * "OneClassEvaluation.jl/src/")
const outputpath = folderpath * "experiments/OneClass/"

using OneClassEvaluation
using ScikitLearn
using ADatasets
using EvalCurves

@sk_import svm: OneClassSVM
@sk_import ensemble: IsolationForest
@sk_import neighbors: LocalOutlierFactor

createOCSVM(γ = "scale") = OneClassSVM(gamma = γ)
createLOF(neighbors::Int = 20) = LocalOutlierFactor(novelty = true, n_neighbors = neighbors, contamination = "auto")
createIF(estimators::Int = 100) = IsolationForest(n_estimators = estimators, behaviour = "new", contamination = "auto")

classificators = [createOCSVM,
                  OneClassEvaluation.KNNAnom,
                  createLOF,
                  createIF
                  ]

parameters = [([[0.01 0.05 0.1 0.5 1. 5. 10. 50. 100.]], ["gamma"]),
              ([[3 4 5], [:kappa :gamma]], ["k", "anomalyMetric"]),
              ([[10 20 50 100]], ["num_neighbors"]),
              ([[50 100 200]], ["num_estimators"])
              ]

cnames = ["OneClassSVM", "kNN", "LocalOutlierFactor", "IsolationForest"]

dataset = "haberman"
if length(ARGS) != 0
    dataset = ARGS[1]
end

loadData(datasetName, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(datasetName, difficulty, dataPath)..., 0.8, "low")
train, test, clusterdness = loadData(dataset, "easy")

normal = collect((train[1][:, train[2] .== 1])')

outp = outputpath * dataset * "/"
mkpath(outp)
result = OneClassEvaluation.runmodels(classificators, parameters, cnames, normal, test, outp)
