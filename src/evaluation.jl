getauc(scores, labels) = ADatasets.auc(ADatasets.roccurve(scores, labels))

gridsearch(f, parameters...) = map(f, Base.product(parameters...))

function runmodel(model, parameters, train, test)
    m = model(parameters...)
    fit!(m, train)
    scores = decision_function(m, test[1])
    println(model)
    println(getauc(scores, test[2] .- 1))
end

runmodels(classificators, parameters, train, test) = map((c, p) -> gridsearch(x -> runmodel(c, x, train, test), p...), classificators, parameters)
