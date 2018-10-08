folderpath = "D:/dev/julia/"
const dataPath = folderpath * "data/loda/public/datasets/numerical"
push!(LOAD_PATH, folderpath, folderpath * "OneClassEvaluation.jl/src/")

using OneClassEvaluation
using ScikitLearn
using ADatasets

@sk_import svm: OneClassSVM

createOCSVM(γ::AbstractFloat) = OneClassSVM(gamma = γ)

classificators = [createOCSVM,
                  OneClassEvaluation.KNNAnom
                  ]

parameters = [[[0.1 0.2]],
              [[3 4 5], [:gamma :delta]]
              ]

loadData(datasetName, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(datasetName, difficulty, dataPath)..., 0.8, "low")
train, test, clusterdness = loadData("abalone", "easy")

normal = collect((train[1][:, train[2] .== 1])')

OneClassEvaluation.runmodels(classificators, parameters, normal, test)
