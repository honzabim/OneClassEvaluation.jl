using OneClassEvaluation
using ScikitLearn
using ADatasets

@sk_import svm: OneClassSVM

createOCSVM(γ::AbstractFloat) = OneClassSVM(gamma = γ)

classificators = [KNNAnom,
                  createOCSVM]

parameters = [[[3 4 5], [:gamma :delta]],
              [[0.1 0.2]]]

loadData(datasetName, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(datasetName, difficulty, dataPath)..., 0.8, "low")
train, test, clusterdness = loadData("abalone", "easy")

normal = train[1][:, train[2] .== 1]

runmodels(classificators, parameters, normal, test)
