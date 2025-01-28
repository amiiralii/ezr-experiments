import random, sys
from ezr import the, DATA, csv, xval, activeLearning, rows, dist
import stats
import regression_baseline as baseline
import csv as cc
import time
from math import sqrt


## Treatments: 
##    Regressors: asIs, Linear, Lasso, Ridge, SVR, lgbm, KNN with Kmeans
##    Number of Samlples: SQRT(N)
##    Acq. functions: No Sampling, Diversity, Random Sampling
##    Main treatment: kmeans (Kmeans++ centroid, neighbor approx method, (1-3-5)NN for prediction)
## Goal: Regression
## Metric: Sum( (real-pred)/sd ) , Sum( abs(real-pred)/sd ), Sum( abs(d2h(real)-d2h(pred))/sd )
def regression(dataset):
    def random_sampling(todo, done):
        random.shuffle(todo)
        return todo

    def diversity_sampling(todo, done):
        # For all candidates in todo
        # Max( Min( dist( candidate , labeled ) ) )
        def dist_to_dones(todo, done):
            dists = [1E+30] * len(todo)
            for i, unlabeled in enumerate(todo):
                for labeled in done:
                    dst = d.dist(labeled, unlabeled)
                    if dst < dists[i]:
                        dists[i] = dst
            i = dists.index(max(dists))
            return dists.index(max(dists))

        # Pick a random candidate from todo 20 times
        # Max( Min( dist( candidate , labeled ) ) )
        def min_max(todo, done):
            dist_picks = {}
            for _ in range(the.fars):
                pick = random.randrange(len(todo))
                dist_picks[pick] = min(d.dist(labeled, todo[pick]) for labeled in done)
            i = max(dist_picks, key= lambda x: dist_picks[x])
            return max(dist_picks, key= lambda x: dist_picks[x])

        #idx = dist_to_dones(todo, done)
        idx = min_max(todo, done)
        todo[0], todo[idx] = todo[idx], todo[0]
        return todo

    def find_leaf(d, clusters, row):
        min_d, idx = 1E+32, 0
        for i, cl in enumerate(clusters):
            distance = d.dist(cl[0], row)
            if distance < min_d:
                min_d = distance
            idx = i
        return clusters[idx]

    t1 = time.time()
    d = DATA().adds(csv(dataset))

    somes, abs_somes, d2h_somes, times  = {}, {}, {}, {}
    for sa in ["non", "RS", "DS", "DS2", "kmeans-1", "kmeans-3", "kmeans-5"]:
        for m in ["asIs", "LR", "LSR", "RR", "SVR", "LGBM", "k1", "k3", "k5"]:
            if "kmeans" in sa:
                if m[0]=="k" and int(m[-1]) <= int(sa[-1]):
                    somes[f"{sa},{m}"] = stats.SOME(txt=f"{sa},{m}")
                    abs_somes[f"{sa},{m}"] = stats.SOME(txt=f"{sa},{m}")
                    d2h_somes[f"{sa},{m}"] = stats.SOME(txt=f"{sa},{m}")
                    times[f"{sa},{m}"] = 0
            elif m[0] != "k": 
                somes[f"{sa},{m}"] = stats.SOME(txt=f"{sa},{m}")
                abs_somes[f"{sa},{m}"] = stats.SOME(txt=f"{sa},{m}")
                d2h_somes[f"{sa},{m}"] = stats.SOME(txt=f"{sa},{m}")
                times[f"{sa},{m}"] = 0


    ## Choosing Samlping Method
    for sampling in ["non", "DS", "DS2", "RS", "kmeans"]:
        
        acq = diversity_sampling if "DS" in sampling else random_sampling if sampling=="RS" else "non"
        sampling_size = int(sqrt(len(d.rows))) 
        random.shuffle(d.rows) 
        ## K-Fold Cross Validation
        for samples,test in xval(d.rows):
            t0 = time.time()
            if sampling in ["DS", "DS2", "RS"]:
                if acq == "DS2":
                    train, _ = d.clone(samples).activeLearning(acquisition = acq, stop = sampling_size, warm = "kmeans")
                else:
                    train, _ = d.clone(samples).activeLearning(acquisition = acq, stop = sampling_size)
            else:
                train = samples
            t1 = time.time()

            for tr in times.keys():
                if sampling in tr: times[tr] += t1-t0
        
            models = {
                "LR"  : baseline.linear, 
                "LSR" : baseline.lasso, 
                "RR"  : baseline.ridge,
                "SVR" : baseline.svr, 
                "LGBM": baseline.lightgbm
            }
            if "kmeans" not in sampling:
                X_train, y_train, X_test, y_test = baseline.prepare(train, test, [c.txt for c in d.cols.all])
                for reg in ["LR", "LSR", "RR", "SVR", "LGBM"]:
                    label = f"{sampling},{reg}"
                    somes[label], abs_somes[label], d2h_somes[label], times[label] = baseline.calc_baseline3(X_train, y_train, X_test, y_test, 
                                                                        somes[label], abs_somes[label], d2h_somes[label], times[label],
                                                                        models[reg], d)
                t0 = time.time()
                dumb_rows = d.clone(random.choices(train, k = 12))
                dumb_mid = dumb_rows.mid()
                t1 = time.time()
                times[f"{sampling},asIs"] += t1-t0
                ## Iterate through each test row
                for want in test:
                    std = d.div()
                    d2h_somes[f"{sampling},asIs"].add( abs(d.d2h(want) - d.d2h(dumb_mid)) )
                    for y in d.cols.y:
                        somes[f"{sampling},asIs"].add( (want[y.at] - dumb_mid[y.at]) / std[y.at])
                        abs_somes[f"{sampling},asIs"].add( abs(want[y.at] - dumb_mid[y.at]) / std[y.at])
                t2 = time.time()
                times[f"{sampling},asIs"] += t2-t1

            else: 
                t0 = time.time()
                cluster_kmeans_1 = d.kmeansplusplus(rows = train, neighbors=1)
                t1 = time.time()
                cluster_kmeans_3 = d.kmeansplusplus(rows = train, neighbors=3)
                t2 = time.time()
                cluster_kmeans_5 = d.kmeansplusplus(rows = train, neighbors=5)
                t3 = time.time()

                for tr in times.keys():
                    if "kmeans-1" in tr : times[tr] += t1-t0
                    elif "kmeans-3" in tr : times[tr] += t2-t1
                    elif "kmeans-5" in tr : times[tr] += t3-t2
            

                ## Iterate through each test row
                for want in test:
                    std = d.div()
                    t0 = time.time()
                    leaf_kmeans_1 = d.clone(find_leaf(d, cluster_kmeans_1, want)).rows
                    t1 = time.time()
                    leaf_kmeans_3 = d.clone(find_leaf(d, cluster_kmeans_3, want)).rows
                    t2 = time.time()
                    leaf_kmeans_5 = d.clone(find_leaf(d, cluster_kmeans_5, want)).rows
                    t3 = time.time()

                    for tr in times.keys():
                        if "kmeans-1" in tr : times[tr] += t1-t0
                        elif "kmeans-3" in tr : times[tr] += t2-t1
                        elif "kmeans-5" in tr : times[tr] += t3-t2

                    ## Regression result per each method
                    for treatment in somes.keys():
                        if "kmeans" in treatment:
                            if "k1" in treatment:
                                kn = 1
                                prediction_rows = leaf_kmeans_1
                            if "k3" in treatment:
                                kn = 3
                                prediction_rows = leaf_kmeans_3
                            if "k5" in treatment:
                                kn = 5
                                prediction_rows = leaf_kmeans_5

                            t1 = time.time()
                            pred = d.predict(want, prediction_rows, k=kn)
                            d2h_somes[treatment].add( abs(d.d2h(want) - d.d2h(pred)) )
                            for y in d.cols.y:
                                abs_somes[treatment].add( abs(want[y.at] - pred[y.at]) / std[y.at])
                                somes[treatment].add( (want[y.at] - pred[y.at]) / std[y.at])
                            t2 = time.time()
                            times[treatment] += t2-t1

    ## Export Run Times
    with open(f"regression10/res/times/{dataset.split('/')[-1]}", 'w') as csv_file:  
        writer = cc.writer(csv_file)
        for i,j in dict(sorted(times.items())).items():
            writer.writerow([i, round(j,2)])

    d2h_res = []
    noabs_res = []
    abs_res = []
    for a,b,c in zip(d2h_somes.values(), somes.values(), abs_somes.values()):
        d2h_res += [a]
        noabs_res += [b]
        abs_res += [c]
    
    return noabs_res, abs_res, d2h_res

dataset = sys.argv[1]
noabs , abs, d2h = regression(dataset)

print(" ++ Standard Residual without abs:")
[stats.report( noabs )]
print("\n\n ++ Standard Residual with abs:")
[stats.report( abs )]
print("\n\n ++ d2h differences:")
[stats.report( d2h )]
