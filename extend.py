import random, sys
from ezr import the, DATA, csv, xval, activeLearning, rows, dist
import stats
import regression_baseline as baseline
import csv as cc
import time
from synthesize import upsampling
from math import sqrt
import pandas as pd

## Treatments: asIs, mid-leaf, 1-3-5 Nearest Neighbors
## Goal: Regression
## Metric: Sum of Absolute Value of Standardized Residuals : Sum( ( (real-pred)/sd ) )
def regression(dataset, repeats):
  loading_time, lr_time, lgbm_time = 0, 0, 0
  asIs_time, midleaf_time, k1_time, k3_time, k5_time = 0, 0, 0, 0, 0

  t1 = time.time()
  d = DATA().adds(csv(dataset))
  loading_time = time.time()-t1
  somes  = []
  lrstat  = stats.SOME(txt='LR')
  lgbmstat  = stats.SOME(txt='LGBM')
  mid1s  = stats.SOME(txt="mid-leaf")
  dumb   = stats.SOME(txt=f"asIs")

  somes += [mid1s]
  somes += [dumb]
  kss = [stats.SOME(txt="k1"), stats.SOME(txt="k3"), stats.SOME(txt="k5")]
  for ks in kss:
    somes += [ks]
  predictions = {}
  predictions["col"] = ["actual", "asIs", "mid-leaf", "k1", "k3", "k5"]

  saving_preds = True
  
  for _ in range(repeats):    
    random.shuffle(d.rows)
    for train,test in xval(d.rows):
      lrstat, lgbmstat, bp , lr_time, lgbm_time= baseline.calc_baseline2(train, test, [c.txt for c in d.cols.all], lrstat, lgbmstat, saving_preds, lr_time, lgbm_time)
      if saving_preds: bl_predictions = bp
      t0 = time.time()
      cluster = d.cluster(train)
      t1 = time.time()
      midleaf_time += t1 - t0
      k1_time += t1 - t0
      k3_time += t1 - t0
      k5_time += t1 - t0
      t1 = time.time()
      dumb_rows = d.clone(random.choices(train, k=the.Stop))
      t2 = time.time()
      asIs_time += t2-t1

      for index, want in enumerate(test):
        t0 = time.time()
        leaf = cluster.leaf(d, want)
        rows = leaf.data.rows
        t1 = time.time()
        midleaf_time += t1 - t0
        k1_time += t1 - t0
        k3_time += t1 - t0
        k5_time += t1 - t0

        mid1  = leaf.data.mid()
        dumb_mid = dumb_rows.mid()

        k_preds  = []
        for k in [1,3,5]:
          t0 = time.time()
          k_preds.append( d.predict(want, rows, k=k) )
          t1 = time.time()
          if k==1: k1_time += t1-t0
          if k==3: k3_time += t1-t0
          if k==5: k5_time += t1-t0
        
        add_all = True
        for k_pred, kstat in zip(k_preds, kss):
          for at,pred in k_pred.items():
            sd = d.cols.all[at].div()
            col = str(index) + '-' + d.cols.all[at].txt
            if (col not in predictions.keys()) and saving_preds:
              predictions[col] = []
            if saving_preds and add_all:
              predictions[col].append(round(want[at],2))
              predictions[col].append(round(dumb_mid[at], 2))
              predictions[col].append(round(mid1[at], 2))
            if saving_preds: predictions[col].append(round(pred, 2))
            
            if add_all:
              mid1s.add( (want[at] - mid1[at])/sd)
              dumb.add( (want[at] - dumb_mid[at])/sd)

            kstat.add(   (want[at] - pred   )/sd)
          add_all = False
      saving_preds = False

  somes += [lrstat]
  somes += [lgbmstat]
  
  ## Export Run Times
  with open(f"reg/res/high-res/times/{dataset.split('/')[-1]}", 'w') as csv_file:  
    writer = cc.writer(csv_file)
    writer.writerow(["loading_time:", round(loading_time,2)])
    writer.writerow(["asIs_time:", round(asIs_time,2)])
    writer.writerow(["lr_time:", round(lr_time,2)])
    writer.writerow(["midleaf_time:", round(midleaf_time,2)])
    writer.writerow(["k1_time:", round(k1_time,2)])
    writer.writerow(["k3_time:", round(k3_time,2)])
    writer.writerow(["k5_time:", round(k5_time,2)])
    writer.writerow(["lgbm_time:", round(lgbm_time,2)])

  ## Export Predictions
  with open(f"reg/res/high-res/predictions/{dataset.split('/')[-1]}", 'w') as csv_file:  
    writer = cc.writer(csv_file)
    for key, value in predictions.items():
       k = [key]
       [k.append(v) for v in value]
       [k.append(v) for v in bl_predictions[key]]
       writer.writerow(k)  
  return somes


## Treatments: asIs, mid-leaf, 1-3-5 Nearest Neighbors, and overpopulated version of each
## Goal: Regression
## Metric: Sum of Absolute Value of Standardized Residuals : Sum( abs( (real-pred)/sd ) )
def regression2(dataset, repeats):
  d = DATA().adds(csv(dataset))
  somes  = []
  mid1s  = stats.SOME(txt="mid-leaf")
  syn_mid1s  = stats.SOME(txt="mid-leaf-syn")
  dumb   = stats.SOME(txt=f"asIs")

  somes += [mid1s]
  somes += [syn_mid1s]
  somes += [dumb]
  kss = [stats.SOME(txt="k1"), stats.SOME(txt="k3"), stats.SOME(txt="k5"), stats.SOME(txt="k1-syn"), stats.SOME(txt="k3-syn"), stats.SOME(txt="k5-syn")]
  for ks in kss:
    somes += [ks]
  res = {}
  res["col"] = ["actual", "asIs", "mid-leaf", "mid-leaf-syn", "k1", "k3", "k5", "k1-syn", "k3-syn", "k5-syn"]

  for rnd in range(repeats):
      
      for train,test in xval(d.rows):
        cluster = d.cluster(train)
        dumb_rows = d.clone(random.choices(train, k=the.Stop))
        for index, want in enumerate(test):
          leaf = cluster.leaf(d, want)
          syn_leaf = upsampling(leaf.data.shuffle(), int(the.Stop))
          rows = leaf.data.rows
          syn_rows = syn_leaf.rows
          gots  = [ d.predict(want, rows, k=k) for k in [1,3,5] ]
          for dp in [d.predict(want, syn_rows, k=k) for k in [1,3,5]]:
            gots.append( dp )

          mid1  = leaf.data.mid()
          dumb_mid = dumb_rows.mid()

          syn_mid1  = syn_leaf.mid()
          
          c=0
          a = {}
          for got, ks in zip(gots, kss):
            for at,got1 in got.items():
              sd = d.cols.all[at].div()
              col = str(index) + '-' + d.cols.all[at].txt
              if (col not in a.keys()) and (rnd==0):
                a[col] = []
              if c==0:
                if rnd==0:
                  a[col].append(round(want[at],2))
                  a[col].append(round(dumb_mid[at], 2))
                  a[col].append(round(mid1[at], 2))
                  a[col].append(round(syn_mid1[at], 2))

                mid1s.add(abs(want[at] - mid1[at])/sd)
                syn_mid1s.add(abs(want[at] - syn_mid1[at])/sd)
                dumb.add(abs(want[at] - dumb_mid[at])/sd)
              if rnd==0:
                a[col].append(round(got1, 2))
              ks.add(  abs(want[at] - got1   )/sd)
            c = 1
            
          if rnd==0:
            res.update(a)
  
  '''    
  with open(f"reg/res/low-res/predictions/{dataset.split('/')[-1]}", 'w') as csv_file:  
    writer = cc.writer(csv_file)
    for key, value in res.items():
       k = [key]
       [k.append(v) for v in value]
       writer.writerow(k)
  '''
  return somes


## Treatments: asIs, mid-leaf, 1-3-5 Nearest Neighbors, each with 10, 30, 50, SQRT(N) cluster population
## Goal: Regression
## Metric: Sum of Absolute Value of Standardized Residuals : Sum( ( (real-pred)/sd ) )
def regression3(dataset, repeats):
  loading_time, lr_time, lgbm_time = 0, 0, 0
  asIs_time, midleaf_time, k1_time, k3_time, k5_time = 0, 0, 0, 0, 0

  t1 = time.time()
  d = DATA().adds(csv(dataset))
  
  somes  = {}
  lrstat = stats.SOME(txt='LR')
  lgbmstat = stats.SOME(txt='LGBM')

  for m in ["asIs", "mid-leaf", "k1", "k3", "k5"]:
    for s in [10, 30, 50, int(sqrt(len(d.rows)))]:
      if s == int(sqrt(len(d.rows))): somes[f"{m},sq"] = stats.SOME(txt=f"{m},sq")
      else: somes[f"{m},{str(s)}"] = stats.SOME(txt=f"{m},{str(s)}")
  
  times = {}
  for m in ["asIs", "mid-leaf", "k1", "k3", "k5"]:
    for s in [10, 30, 50, int(sqrt(len(d.rows)))]:
      if s == int(sqrt(len(d.rows))): times[f"{m},sq"] = 0
      else: times[f"{m},{str(s)}"] = 0

  saving_preds = False
  
  ## Repeating Experiment
  for _ in range(repeats):
    random.shuffle(d.rows)
    ## K-Fold Cross Validation
    for train,test in xval(d.rows):
      lrstat, lgbmstat, bp , lr_time, lgbm_time= baseline.calc_baseline2(train, test, [c.txt for c in d.cols.all], lrstat, lgbmstat, saving_preds, lr_time, lgbm_time)
      if saving_preds: bl_predictions = bp
      ## different leaf size
      for stp in [10, 30, 50, int(sqrt(len(d.rows)))]:
        t0 = time.time()
        cluster = d.cluster(train, stop = stp)
        t1 = time.time()
        dumb_rows = d.clone(random.choices(train, k=stp))
        dumb_mid = dumb_rows.mid()
        t2 = time.time()

        for tr in times.keys():
          if "asIs" in tr: times[tr] += t2-t1
          else: times[tr] += t1-t0

        ## Iterate through each test row
        for want in test:
          std = d.div()
          leaf = cluster.leaf(d, want)
          rows = leaf.data.rows
          mid1  = leaf.data.mid()
          ## Regression result per each method
          for treatment in somes.keys():
            if "asIs" in treatment:
              t1 = time.time()
              for y in d.cols.y:
                somes[treatment].add( (want[y.at] - dumb_mid[y.at]) / std[y.at])
              t2 = time.time()
              times[treatment] += t2-t1

            if "mid-leaf" in treatment:
              t1 = time.time()
              for y in d.cols.y:
                somes[treatment].add( (want[y.at] - mid1[y.at]) / std[y.at])
              t2 = time.time()
              times[treatment] += t2-t1

            if "k1" in treatment:
              t1 = time.time()
              pred = d.predict(want, rows, k=1)
              for y in d.cols.y:
                somes[treatment].add( (want[y.at] - pred[y.at]) / std[y.at])
              t2 = time.time()
              times[treatment] += t2-t1

            if "k3" in treatment:
              t1 = time.time()
              pred = d.predict(want, rows, k=3)
              for y in d.cols.y:
                somes[treatment].add( (want[y.at] - pred[y.at]) / std[y.at])
              t2 = time.time()
              times[treatment] += t2-t1

            if "k5" in treatment:
              t1 = time.time()
              pred = d.predict(want, rows, k=5)
              for y in d.cols.y:
                somes[treatment].add( (want[y.at] - pred[y.at]) / std[y.at])   
              t2 = time.time()
              times[treatment] += t2-t1 


  ## Export Run Times
  with open(f"reg3/res/low-res/times/{dataset.split('/')[-1]}", 'w') as csv_file:  
      writer = cc.writer(csv_file)
      for i,j in dict(sorted(times.items())).items():
        writer.writerow([i, round(j,2)])
      
  
  if saving_preds:
    ## Export Predictions
    with open(f"reg/res/high-res/predictions/{dataset.split('/')[-1]}", 'w') as csv_file:  
      writer = cc.writer(csv_file)
      for key, value in predictions.items():
        k = [key]
        [k.append(v) for v in value]
        [k.append(v) for v in bl_predictions[key]]
        writer.writerow(k)  
  
  res = []
  for m in somes.values():
    res += [m]
  res += [lrstat]
  res += [lgbmstat]
  return res


## Treatments: different Acquisition functions
## Goal: Sampling and Regression
## Metric: Sum of Absolute Value of Standardized Residuals : Sum( ( (real-pred)/sd ) )
def regression4(dataset, repeats):
  def random_sampling(todo:rows, done:rows) -> rows:
    random.shuffle(todo)
    return todo
  
  def greedy_sampling(todo:rows, done:rows) -> rows:
    min_dists = dist_to_dones(todo, done)
    idx = min_dists.index(max(min_dists))
    todo[0], todo[idx] = todo[idx], todo[0]
    return todo
      
  def uncertainty_sampling(todo:rows, done:rows, adopt) -> rows:
    errors = uncertainty(todo, done, [c.txt for c in d.cols.all], adopt)
    idx = max(errors, key= lambda x: errors[x])
    todo[0], todo[idx] = todo[idx], todo[0]
    return todo

  def uncertainty(test, train, cols, regressor):
    X_train, y_train, X_test, y_test = reg_prepare(train, test, cols)

    sdvs = {target_column : y_train[target_column].std() for target_column in y_train.columns}
    errors = {idx : 0 for idx in range(len(y_test))}

    for target_column in y_train.columns:
        y_pred_lr = regressor( X_train, y_train[target_column], X_test)
        for idx in range(len(y_pred_lr)):
            errors[idx] += abs(y_test[target_column].iloc[idx] - y_pred_lr[idx])/(sdvs[target_column]+ 1E-30)
    return errors
  
  def dist_to_dones(todo:rows, done:rows):
    dists = [1E+30] * len(todo)
    for i, unlabeled in enumerate(todo):
      for labeled in done:
        dst = d.dist(labeled, unlabeled)
        if dst < dists[i]:
          dists[i] = dst
    return dists

  def reg_prepare(train, test, cols):

    train_df = pd.DataFrame(train, columns=cols)
    test_df = pd.DataFrame(test, columns=cols)

    X_train, y_train = baseline.xy(train_df)
    X_test, y_test = baseline.xy(test_df)

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_test = y_test.reindex(columns=y_train.columns, fill_value=0)

    return X_train, y_train, X_test, y_test

  def reg_predict(regressor, stat, X_train, y_train, X_test, y_test):
    for target_column in y_train.columns:
        y_pred_lr = regressor( X_train, y_train[target_column], X_test)
        sdv = y_train[target_column].std()
        for idx in range(len(y_test)):
            stat.add( (y_test[target_column].iloc[idx] - y_pred_lr[idx])/sdv)
    return stat
  
  d = DATA().adds(csv(dataset))
  map_name = {random_sampling:'RS', uncertainty_sampling: 'US', greedy_sampling: 'GS',
                baseline.linear: 'LR', baseline.ridge: 'RR', baseline.random_forest: 'RF', baseline.lightgbm:'LGBM'}
  somes  = []
  labeled_size = 0.2 * 0.8 * len(d.rows)

  ## Repeating experiment for Statistical validation
  for _ in range(repeats):
    ## K-Fold Cross Validation
    for train,test in xval(d.rows):
      ## Iterating over acquisition functions
      # passive methods
      for acq_func in [greedy_sampling, random_sampling]:
        samples, _ = d.clone(train).activeLearning(acquisition=acq_func, stop = labeled_size)
        X_train, y_train, X_test, y_test = reg_prepare(samples, test, [c.txt for c in d.cols.all])
        # Iterating over regressors
        for regressor in [baseline.linear, baseline.ridge, baseline.random_forest, baseline.lightgbm]:
          trt_name = f'{map_name[acq_func]},{map_name[regressor]}'
          treatment = [trt for trt in somes if trt.txt == trt_name]
          if not treatment:
            treatment = stats.SOME(txt=trt_name)
            somes += [treatment]
          else:
            treatment = treatment[0]
          treatment = reg_predict(regressor, treatment, X_train, y_train, X_test, y_test)
      
      # adoptive methods
      for acq_func in [uncertainty_sampling]:
        # Iterating over regressors
        for regressor in [baseline.linear, baseline.ridge, baseline.random_forest, baseline.lightgbm]:
          samples, _ = d.clone(train).activeLearning(acquisition=acq_func, adopt = regressor, stop= labeled_size)
          X_train, y_train, X_test, y_test = reg_prepare(samples, test, [c.txt for c in d.cols.all])

          trt_name = f'{map_name[acq_func]},{map_name[regressor]}'
          treatment = [trt for trt in somes if trt.txt == trt_name]
          if not treatment:
            treatment = stats.SOME(txt=trt_name)
            somes += [treatment]
          else:
            treatment = treatment[0]
          treatment = reg_predict(regressor, treatment, X_train, y_train, X_test, y_test)
  return somes


## Treatments: asIs, mid-leaf, 1-3-5 Nearest Neighbors, each with SQRT(N) samples produced GS,RS
## Goal: Regression
## Metric: Sum of Absolute Value of Standardized Residuals : Sum( ( (real-pred)/sd ) )
def regression5(dataset, repeats):
  def random_sampling(todo, done):
    random.shuffle(todo)
    return todo
  
  def greedy_sampling(todo, done):
    min_dists = dist_to_dones(todo, done)
    idx = min_dists.index(max(min_dists))
    todo[0], todo[idx] = todo[idx], todo[0]
    return todo
  
  def dist_to_dones(todo, done):
    dists = [1E+30] * len(todo)
    for i, unlabeled in enumerate(todo):
      for labeled in done:
        dst = d.dist(labeled, unlabeled)
        if dst < dists[i]:
          dists[i] = dst
    return dists

  t1 = time.time()
  d = DATA().adds(csv(dataset))
  
  somes  = {}
  for sa in ["RS", "GS"]:
    for m in ["asIs", "mid-leaf", "k1", "k3", "k5", "LR", "LGBM"]:
      for s in [int(sqrt(len(d.rows)))]:
        if s == int(sqrt(len(d.rows))): somes[f"{m},{sa}"] = stats.SOME(txt=f"{m},{sa}")
        else: somes[f"{m},{str(s)}"] = stats.SOME(txt=f"{m},{sa}")
  
  times = {}
  for sa in ["RS", "GS"]:
    for m in ["asIs", "mid-leaf", "k1", "k3", "k5", "LR", "LGBM"]:
      for s in [int(sqrt(len(d.rows)))]:
        if s == int(sqrt(len(d.rows))): times[f"{m},{sa}"] = 0
        else: times[f"{m},{sa}"] = 0

  saving_preds = False
  
  ## Repeating Experiment
  for _ in range(repeats):
    ## Choosing Samlping Method
    for sampling in [greedy_sampling, random_sampling]:
      acq = "GS" if sampling==greedy_sampling else "RS"
      ## Choosing sampling rate
      for stp in [int(sqrt(len(d.rows)))]:
        random.shuffle(d.rows) 
        ## K-Fold Cross Validation
        for samples,test in xval(d.rows):
          t0 = time.time()
          train, _ = d.clone(samples).activeLearning(acquisition = sampling, stop = stp)
          t1 = time.time()

          for tr in times.keys():
              if acq in tr: times[tr] += t1-t0

          lrlabel = f"LR,{acq}"
          lgbmlabel = f"LGBM,{acq}"

          somes[lrlabel], somes[lgbmlabel], _ , times[lrlabel], times[lgbmlabel] = baseline.calc_baseline2(train, test, [c.txt for c in d.cols.all], somes[lrlabel], somes[lgbmlabel], saving_preds, times[lrlabel], times[lgbmlabel])

          t0 = time.time()
          cluster = d.cluster(train, stop = 12)
          t1 = time.time()
          dumb_rows = d.clone(random.choices(train, k=12))
          dumb_mid = dumb_rows.mid()
          t2 = time.time()

          for tr in times.keys():
            if "asIs" in tr and acq in tr: times[tr] += t2-t1
            elif acq in tr: times[tr] += t1-t0

          ## Iterate through each test row
          for want in test:
            std = d.div()
            leaf = cluster.leaf(d, want)
            rows = leaf.data.rows
            mid1  = leaf.data.mid()
            ## Regression result per each method
            for treatment in somes.keys():
              if "asIs" in treatment and acq in treatment:
                t1 = time.time()
                for y in d.cols.y:
                  somes[treatment].add( (want[y.at] - dumb_mid[y.at]) / std[y.at])
                t2 = time.time()
                times[treatment] += t2-t1

              if "mid-leaf" in treatment and acq in treatment:
                t1 = time.time()
                for y in d.cols.y:
                  somes[treatment].add( (want[y.at] - mid1[y.at]) / std[y.at])
                t2 = time.time()
                times[treatment] += t2-t1

              if "k1" in treatment and acq in treatment:
                t1 = time.time()
                pred = d.predict(want, rows, k=1)
                for y in d.cols.y:
                  somes[treatment].add( (want[y.at] - pred[y.at]) / std[y.at])
                t2 = time.time()
                times[treatment] += t2-t1

              if "k3" in treatment and acq in treatment:
                t1 = time.time()
                pred = d.predict(want, rows, k=3)
                for y in d.cols.y:
                  somes[treatment].add( (want[y.at] - pred[y.at]) / std[y.at])
                t2 = time.time()
                times[treatment] += t2-t1

              if "k5" in treatment and acq in treatment:
                t1 = time.time()
                pred = d.predict(want, rows, k=5)
                for y in d.cols.y:
                  somes[treatment].add( (want[y.at] - pred[y.at]) / std[y.at])   
                t2 = time.time()
                times[treatment] += t2-t1 


  ## Export Run Times
  with open(f"reg5/res/low-res/times/{dataset.split('/')[-1]}", 'w') as csv_file:  
      writer = cc.writer(csv_file)
      for i,j in dict(sorted(times.items())).items():
        writer.writerow([i, round(j,2)])
      
  
  res = []
  for m in somes.values():
    res += [m]
  return res


## Treatments: 
##    Regressors: asIs, mid-leaf, 1-3-5 Nearest Neighbors, Linear, Lasso, Ridge, SVR, lgbm
##    Number of Samlples: SQRT(N)
##    Acq. functions: No Sampling, Diversity, Random Sampling
## Goal: Regression
## Metric: Sum of Absolute Value of Standardized Residuals : Sum( ( (real-pred)/sd ) )
def regression6(dataset, repeats):
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

  t1 = time.time()
  d = DATA().adds(csv(dataset))
  
  somes  = {}
  for sa in ["non", "RS", "DS"]:
    for m in ["asIs", "mid-leaf", "k1", "k3", "k5", "LR", "LSR", "RR", "SVR", "LGBM"]:
      somes[f"{sa},{m}"] = stats.SOME(txt=f"{sa},{m}")
  
  times = {}
  for sa in ["non", "RS", "DS"]:
    for m in ["asIs", "mid-leaf", "k1", "k3", "k5", "LR", "LSR", "RR", "SVR", "LGBM"]:
      times[f"{sa},{m}"] = 0

  ## Repeating Experiment
  for _ in range(repeats):
    ## Choosing Samlping Method
    for sampling in ["non", diversity_sampling, random_sampling]:
      acq = "DS" if sampling==diversity_sampling else "RS" if sampling==random_sampling else "non"
      ## Choosing sampling rate
      for stp in [int(sqrt(len(d.rows)))]:
        random.shuffle(d.rows) 
        ## K-Fold Cross Validation
        for samples,test in xval(d.rows):
          t0 = time.time()
          if acq != "non":
            train, _ = d.clone(samples).activeLearning(acquisition = sampling, stop = stp)
          else:
            train = samples
          t1 = time.time()

          for tr in times.keys():
              if acq in tr: times[tr] += t1-t0
        
          models = {
              "LR"  : baseline.linear, 
              "LSR" : baseline.lasso, 
              "RR"  : baseline.ridge,
              "SVR" : baseline.svr, 
              "LGBM": baseline.lightgbm
          }
          X_train, y_train, X_test, y_test = baseline.prepare(train, test, [c.txt for c in d.cols.all])
          for reg in ["LR", "LSR", "RR", "SVR", "LGBM"]:
            label = f"{acq},{reg}"
            somes[label], times[label] = baseline.calc_baseline3(X_train, y_train, X_test, y_test, somes[label], times[label], models[reg])


          t0 = time.time()
          cluster = d.cluster(train, stop = 12)
          t1 = time.time()
          dumb_rows = d.clone(random.choices(train, k = 12))
          dumb_mid = dumb_rows.mid()
          t2 = time.time()

          for tr in times.keys():
            if "asIs" in tr and acq in tr: times[tr] += t2-t1
            elif acq in tr: times[tr] += t1-t0

          ## Iterate through each test row
          for want in test:
            std = d.div()
            leaf = cluster.leaf(d, want)
            rows = leaf.data.rows
            mid1  = leaf.data.mid()
            ## Regression result per each method
            for treatment in somes.keys():
              if "asIs" in treatment and acq in treatment:
                t1 = time.time()
                for y in d.cols.y:
                  somes[treatment].add( (want[y.at] - dumb_mid[y.at]) / std[y.at])
                t2 = time.time()
                times[treatment] += t2-t1

              if "mid-leaf" in treatment and acq in treatment:
                t1 = time.time()
                for y in d.cols.y:
                  somes[treatment].add( (want[y.at] - mid1[y.at]) / std[y.at])
                t2 = time.time()
                times[treatment] += t2-t1

              if "k1" in treatment and acq in treatment:
                t1 = time.time()
                pred = d.predict(want, rows, k=1)
                for y in d.cols.y:
                  somes[treatment].add( (want[y.at] - pred[y.at]) / std[y.at])
                t2 = time.time()
                times[treatment] += t2-t1

              if "k3" in treatment and acq in treatment:
                t1 = time.time()
                pred = d.predict(want, rows, k=3)
                for y in d.cols.y:
                  somes[treatment].add( (want[y.at] - pred[y.at]) / std[y.at])
                t2 = time.time()
                times[treatment] += t2-t1

              if "k5" in treatment and acq in treatment:
                t1 = time.time()
                pred = d.predict(want, rows, k=5)
                for y in d.cols.y:
                  somes[treatment].add( (want[y.at] - pred[y.at]) / std[y.at])   
                t2 = time.time()
                times[treatment] += t2-t1 


  ## Export Run Times
  with open(f"reg6/res/times/{dataset.split('/')[-1]}", 'w') as csv_file:  
      writer = cc.writer(csv_file)
      for i,j in dict(sorted(times.items())).items():
        writer.writerow([i[1:-1], round(j,2)])
  
  res = []
  for m in somes.values():
    res += [m]
  
  return res


## Treatments: 
##    Regressors: asIs, mid-leaf, 1-3-5 Nearest Neighbors, Linear, Lasso, Ridge, SVR, lgbm
##    Number of Samlples: SQRT(N)
##    Acq. functions: No Sampling, Diversity, Random Sampling
## Goal: Regression
## Metric: Sum of Absolute Value of Standardized Residuals : Sum( ( (real-pred)/sd ) )
def regression7(dataset, repeats):
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
  
  somes  = {}
  for sa in ["non"]:
    for m in ["asIs", "mid-leaf", "k1", "k3", "k5"]:
      somes[f"{sa},{m}"] = stats.SOME(txt=f"{sa},{m}")
  
  times = {}
  for sa in ["non"]:
    for m in ["asIs", "mid-leaf", "k1", "k3", "k5"]:
      times[f"{sa},{m}"] = 0

  ## Repeating Experiment
  for _ in range(repeats):
    ## Choosing Samlping Method
    ##for sampling in ["non", diversity_sampling, random_sampling]:
    for sampling in ["non"]:
      acq = "DS" if sampling==diversity_sampling else "RS" if sampling==random_sampling else "non"
      ## Choosing sampling rate
      for stp in [int(sqrt(len(d.rows)))]:
        random.shuffle(d.rows) 
        ## K-Fold Cross Validation
        for samples,test in xval(d.rows):
          t0 = time.time()
          if acq != "non":
            train, _ = d.clone(samples).activeLearning(acquisition = sampling, stop = stp)
          else:
            train = samples
          t1 = time.time()

          for tr in times.keys():
              if acq in tr: times[tr] += t1-t0
        
          models = {
              "LR"  : baseline.linear, 
              "LSR" : baseline.lasso, 
              "RR"  : baseline.ridge,
              "SVR" : baseline.svr, 
              "LGBM": baseline.lightgbm
          }
          #X_train, y_train, X_test, y_test = baseline.prepare(train, test, [c.txt for c in d.cols.all])
          #for reg in ["LR", "LSR", "RR", "SVR", "LGBM"]:
          #  label = f"{acq},{reg}"
          #  somes[label], times[label] = baseline.calc_baseline3(X_train, y_train, X_test, y_test, somes[label], times[label], models[reg])


          t0 = time.time()
          ## This is where I had to add kmeans algorithm instead of d.cluster()
          #cluster = d.cluster(train, stop = 12)
          clusters = d.kmeansplusplus(rows = train, leaf_size = 12)
          t1 = time.time()
          dumb_rows = d.clone(random.choices(train, k = 12))
          dumb_mid = dumb_rows.mid()
          t2 = time.time()

          for tr in times.keys():
            if "asIs" in tr and acq in tr: times[tr] += t2-t1
            elif acq in tr: times[tr] += t1-t0

          ## Iterate through each test row
          for want in test:
            std = d.div()
            #leaf = cluster.leaf(d, want)
            leaf = d.clone(find_leaf(d, clusters, want))
            #rows = leaf.data.rows
            rows = leaf.rows
            #mid1  = leaf.data.mid()
            mid1  = leaf.mid()
            
            ## Regression result per each method
            for treatment in somes.keys():
              if "asIs" in treatment and acq in treatment:
                t1 = time.time()
                for y in d.cols.y:
                  somes[treatment].add( (want[y.at] - dumb_mid[y.at]) / std[y.at])
                t2 = time.time()
                times[treatment] += t2-t1

              if "mid-leaf" in treatment and acq in treatment:
                t1 = time.time()
                for y in d.cols.y:
                  somes[treatment].add( (want[y.at] - mid1[y.at]) / std[y.at])
                t2 = time.time()
                times[treatment] += t2-t1

              if "k1" in treatment and acq in treatment:
                t1 = time.time()
                pred = d.predict(want, rows, k=1)
                for y in d.cols.y:
                  somes[treatment].add( (want[y.at] - pred[y.at]) / std[y.at])
                t2 = time.time()
                times[treatment] += t2-t1

              if "k3" in treatment and acq in treatment:
                t1 = time.time()
                pred = d.predict(want, rows, k=3)
                for y in d.cols.y:
                  somes[treatment].add( (want[y.at] - pred[y.at]) / std[y.at])
                t2 = time.time()
                times[treatment] += t2-t1

              if "k5" in treatment and acq in treatment:
                t1 = time.time()
                pred = d.predict(want, rows, k=5)
                for y in d.cols.y:
                  somes[treatment].add( (want[y.at] - pred[y.at]) / std[y.at])   
                t2 = time.time()
                times[treatment] += t2-t1 


  ## Export Run Times
  with open(f"reg7/res/times/{dataset.split('/')[-1]}", 'w') as csv_file:  
      writer = cc.writer(csv_file)
      for i,j in dict(sorted(times.items())).items():
        writer.writerow([i.replace("non","kmeans"), round(j,2)])
  
  res = []
  for m in somes.values():
    res += [m]
  for r in res:
    r.txt = r.txt.replace("non","kmeans")
  
  return res




dataset = sys.argv[1]
repeats = 20
[stats.report( regression7(dataset, repeats) ) ]
