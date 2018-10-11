getauc(scores, labels) = ADatasets.auc(ADatasets.roccurve(scores, labels)...)

gridsearch(f, parameters...) = vcat(map(f, Base.product(parameters...))...)

function precision_at_k(scores, labels, k::Int)
    isort = sortperm(scores, rev = true)
    return mean(labels[isort][1:k])
end

function threshold_at_fpr(scores, labels, fpr)
    descendingidx = sortperm(scores, rev = true)
    scores = scores[descendingidx]
    labels = labels[descendingidx]

    distincvalueidx = findall(diff(scores) .!= 0)
    thresholdidx = vcat(distincvalueidx, length(labels))

    tps = cumsum(labels)[thresholdidx]
    fps = thresholdidx .- tps
    fps = fps ./ fps[end]

    thresholds = scores[thresholdidx]

    ids = fpr .>= fps
    lastsmaller = sum(ids)
    if lastsmaller == 0
        @warn "No score to estimate lower FPR than $(fps[1])"
        return NaN # thresholds[1]
    elseif lastsmaller == length(fps)
        @warn "No score to estimate higher FPR than $(fps[end])"
        return NaN # thresholds[end]
    end

    return (thresholds[lastsmaller] + thresholds[lastsmaller + 1]) / 2
end

function estimate_volume_for_threshold(scorefunc, threshold, mins, maxs, samples::Int = 10000)
    s = rand(samples, length(mins))
    s .*= (maxs .- mins)
    s .+= mins
    scores = scorefunc(s)
    # println(scores)
    # println(threshold)
    return count(scores .<= threshold) / samples
end

function estimate_volume_for_threshold_discrete(scorefunc, threshold, values, samples::Int = 10000)
    println("Computing volume from discrete samples")
    s = hcat(map(v -> v[rand(1:length(v), samples)], values)...)
    scores = scorefunc(s)
    # println(scores)
    # println(threshold)
    return count(scores .<= threshold) / samples
end

mc_volume_estimate(scorefunc, threshold, iter::Int, mins, maxs) = mc_volume_estimate(scorefunc, threshold, estimate_volume_for_threshold, iter, mins, maxs)
mc_volume_estimate(scorefunc, threshold, iter::Int, values) = mc_volume_estimate(scorefunc, threshold, estimate_volume_for_threshold_discrete, iter, values)
function mc_volume_estimate(scorefunc, threshold, estimatefunc, iter::Int, pars...)
    volume = 0
    for i in 1:iter
        volume += estimatefunc(scorefunc, threshold, pars...)
    end
    return volume / iter
end

function runmodel(model, parameters, parnames, name, train, test)
    dfs = []
    for iteration in 1:1
        println("Iteration: $iteration")
        df = DataFrame()
        df[:name] = name
        df[:iteration] = iteration
        foreach((pname, pval) -> df[Symbol(pname)] = pval, parnames, parameters)

        m = model(parameters...)
        ScikitLearn.fit!(m, train)
        scores = .- decision_function(m, collect(test[1]'))[:]
        labels = test[2] .- 1

        fpr, tpr = EvalCurves.roccurve(scores, labels)
        auc = getauc(scores, labels)
        df[:auc] = auc
        tpr_at_5 = EvalCurves.tpr_at_fpr(fpr, tpr, 0.05)
        df[:tpr_at_5] = tpr_at_5
        auc_at_5 = EvalCurves.auc_at_p(fpr, tpr, 0.05)
        df[:auc_at_5] = auc_at_5
        prec_at_k = precision_at_k(scores, labels, 10)
        df[:prec_at_k] = prec_at_k
        tshld = threshold_at_fpr(scores, labels, 0.05)
        df[:tshld05] = tshld

        alldata = vcat(train, collect(test[1]'))
        volume = 0
        if length(unique(alldata[:, 1])) / size(alldata, 1) <= 0.1
            values = []
            for i in 1:size(alldata, 2)
                push!(values, unique(alldata[:, i]))
            end
            volume = mc_volume_estimate(x -> .- decision_function(m, x)[:], threshold_at_fpr(scores, labels, 0.05), 10, values)
        else
            volume = mc_volume_estimate(x -> .- decision_function(m, x)[:], threshold_at_fpr(scores, labels, 0.05), 10, minimum(alldata, dims = 1), maximum(alldata, dims = 1))
        end
        df[:volume05] = volume

        # Console output
        println(name)
        for i in 1:length(parameters)
            println("$(parnames[i]) = $(parameters[i])")
        end
        println("AUC = $auc")
        push!(dfs, df)
    end
    return vcat(dfs...)
end

function runandsave(classifier, parameters, name, train, test, folder)
    result = gridsearch(pars -> runmodel(classifier, pars, parameters[2], name, train, test), parameters[1]...)
    CSV.write(folder * name * ".csv", result)
    return result
end

runmodels(classificators, parameters, names, train, test, folder) = map((c, p, n) -> runandsave(c, p, n, train, test, folder), classificators, parameters, names)
