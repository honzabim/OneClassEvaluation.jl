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
        warn("No score to estimate lower FPR than $(fps[1])")
        return NaN # thresholds[1]
    elseif lastsmaller == length(fps)
        warn("No score to estimate higher FPR than $(fps[end])")
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

function mc_volume_estimate(scorefunc, threshold, mins, maxs, ϵ, iter::Int = 0)
    if iter != 0
        volume = 0
        for i in 1:iter
            volume += estimate_volume_for_threshold(scorefunc, threshold, mins, maxs)
        end
        return volume / iter
    else
        Δ = 1
        i = 1
        volume = estimate_volume_for_threshold(scorefunc, threshold, mins, maxs)
        while Δ > ϵ && i < 1000000 # safety so it stops at some point
            newvolume = (volume * i + estimate_volume_for_threshold(scorefunc, threshold, mins, maxs)) / (i + 1)
            Δ = abs(newvolume - volume)
            i += 1
            volume = newvolume
        end
        return volume
    end
end

function runmodel(model, parameters, parnames, name, train, test)

    df = DataFrame()
    df[:name] = name
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
    volume = mc_volume_estimate(x -> .- decision_function(m, x)[:], threshold_at_fpr(scores, labels, 0.05), minimum(alldata, dims = 1), maximum(alldata, dims = 1), 0, 10)
    df[:volume05] = volume
    volume = mc_volume_estimate(x -> .- decision_function(m, x)[:], threshold_at_fpr(scores, labels, 0.1), minimum(alldata, dims = 1), maximum(alldata, dims = 1), 0, 10)
    df[:volume10] = volume
    volume = mc_volume_estimate(x -> .- decision_function(m, x)[:], threshold_at_fpr(scores, labels, 0.5), minimum(alldata, dims = 1), maximum(alldata, dims = 1), 0, 10)
    df[:volume50] = volume

    # Console output
    println(name)
    for i in 1:length(parameters)
        println("$(parnames[i]) = $(parameters[i])")
    end
    println("AUC = $auc")

    return df
end

function runandsave(classifier, parameters, name, train, test, folder)
    result = gridsearch(pars -> runmodel(classifier, pars, parameters[2], name, train, test), parameters[1]...)
    CSV.write(folder * name * ".csv", result)
    return result
end

runmodels(classificators, parameters, names, train, test, folder) = map((c, p, n) -> runandsave(c, p, n, train, test, folder), classificators, parameters, names)
