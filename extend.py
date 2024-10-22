import random, sys
from ezr import the, DATA, csv, xval
import stats
import regression_baseline as baseline
import csv as cc
import time
from synthesize import upsampling
from math import sqrt


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


## Treatments: asIs, mid-leaf, 1-3-5 Nearest Neighbors, each with 10, 30, 50, SQRT(N) clusters
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


dataset = sys.argv[1]
repeats = 1
[stats.report( regression3(dataset, repeats) ) ]
