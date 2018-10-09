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

parameters = [[[0.1 0.2]],
              [[3 4 5], [:kappa :gamma :delta]],
              [[10 20 50]],
              [[50 100 200]]
              ]

loadData(datasetName, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(datasetName, difficulty, dataPath)..., 0.8, "low")
train, test, clusterdness = loadData("abalone", "easy")

normal = collect((train[1][:, train[2] .== 1])')

OneClassEvaluation.runmodels(classificators, parameters, normal, test)
