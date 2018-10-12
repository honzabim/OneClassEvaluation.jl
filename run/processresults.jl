using CSV
using DataFrames
using Plots
using Measures

plotly()

folderpath = "D:/dev/julia/"
# folderpath = "/home/bimjan/dev/julia/"
const dataPath = folderpath * "experiments/OneClass/"
datasets = readdir(dataPath)
cnames = ["OneClassSVM", "kNN", "LocalOutlierFactor", "IsolationForest"]
perfmetrics = [:auc, :auc_at_5, :tpr_at_5, :prec_at_k]
pmdfs = [[], [], [], []]

getpmetrics(df, row) = df[row, vcat(perfmetrics, :volume05)]

function allperdataset(d)
    all = []
    for c in cnames
       allresults = CSV.read(dataPath * d * "/" * c * ".csv")

       # filter all that have threshold for FPR 0.05 = NaN as there is no reliable way to estimate that threshold and therefore the other metrics
       allresults = allresults[isnan.(allresults[:tshld05]) .== false, :]
       allresults[:dataset] = d
       allresults[:classifier] = c
       push!(all, allresults[vcat(:dataset, :classifier, perfmetrics..., :volume05)])
   end
   vcat(all...)
end

function getmaxformetrics()
    for d in datasets
        for c in cnames
            allresults = CSV.read(dataPath * d * "/" * c * ".csv")

            # filter all that have threshold for FPR 0.05 = NaN as there is no reliable way to estimate that threshold and therefore the other metrics
            allresults = allresults[isnan.(allresults[:tshld05]) .== false, :]

            for i in 1:length(perfmetrics)
                pm = perfmetrics[i]
                pmdf = pmdfs[i]
                newrow = DataFrame(dataset = d, classifier = c)
                newrow = hcat(newrow, getpmetrics(allresults, argmax(allresults[pm])))
                push!(pmdf, newrow)
            end
        end
    end

    aucdf, aucat5df, trpat5df, precatkdf = map(vec -> vcat(vec...), pmdfs)
end

function plotall(x, y)
    scatters = []
    for d in datasets
        data = allperdataset(d)
        s = scatter(data[x], data[y], title = d, xlabel = "$x", ylabel = "$y", margin = 10mm)
        push!(scatters, s)
    end
    display(plot(scatters..., layout=(3, 4), size = (1600, 900), legend = false))
end

function plotallcombinations()
    p = vcat(perfmetrics, :volume05)
    for i in 1:length(p)
        for j in (i + 1):length(p)
            plotall(p[j], p[i])
        end
    end
end
