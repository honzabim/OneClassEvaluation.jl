folderpath = "D:/dev/julia/"
const dataPath = folderpath * "data/loda/public/datasets/numerical"
push!(LOAD_PATH, folderpath, folderpath * "OneClassEvaluation.jl/src/")

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

parameters = [([["scale"]], ["gamma"]),
              ([[3 4 5], [:kappa :gamma :delta]], ["k", "anomalyMetric"]),
              ([[10 20 50]], ["num_neighbors"]),
              ([[50 100 200]], ["num_estimators"])
              ]

cnames = ["OneClassSVM", "kNN", "LocalOutlierFactor", "IsolationForest"]

loadData(datasetName, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(datasetName, difficulty, dataPath)..., 0.8, "low")
train, test, clusterdness = loadData("pendigits", "easy")

normal = collect((train[1][:, train[2] .== 1])')

result = OneClassEvaluation.runmodels(classificators, parameters, cnames, normal, test)
