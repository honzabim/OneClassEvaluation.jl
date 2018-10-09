getauc(scores, labels) = ADatasets.auc(ADatasets.roccurve(scores, labels)...)

gridsearch(f, parameters...) = vcat(map(f, Base.product(parameters...))...)

function runmodel(model, parameters, parnames, name, train, test)

    df = DataFrame()
    df[:name] = name
    foreach((pname, pval) -> df[Symbol(pname)] = pval, parnames, parameters)

    m = model(parameters...)
    ScikitLearn.fit!(m, train)
    scores = .- decision_function(m, collect(test[1]'))[:]

    auc = getauc(scores, test[2] .- 1)

    df[:auc] = auc

    # Console output
    println(name)
    for i in 1:length(parameters)
        println("$(parnames[i]) = $(parameters[i])")
    end
    println("AUC = $auc")

    return df
end

runmodels(classificators, parameters, names, train, test) = map((c, p, n) -> gridsearch(pars -> runmodel(c, pars, p[2], n, train, test), p[1]...), classificators, parameters, names)
