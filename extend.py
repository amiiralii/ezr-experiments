import random, sys
from ezr import the, DATA, csv, xval
import stats
import regression_baseline as baseline
import csv as cc
import time
from synthesize import upsampling


## Treatments: asIs, mid-leaf, 1-2-3 Nearest Neighbors
## Goal: Regression
## Metric: Sum of Absolute Value of Standardized Residuals : Sum( abs( (real-pred)/sd ) )
def regression(dataset, repeats):
  d = DATA().adds(csv(dataset))
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
      lrstat, lgbmstat, bp = baseline.calc_baseline2(train, test, [c.txt for c in d.cols.all], lrstat, lgbmstat, saving_preds)
      if saving_preds: bl_predictions = bp

      cluster = d.cluster(train)
      dumb_rows = d.clone(random.choices(train, k=the.Stop))
      for index, want in enumerate(test):
        leaf = cluster.leaf(d, want)
        rows = leaf.data.rows
        mid1  = leaf.data.mid()
        dumb_mid = dumb_rows.mid()

        k_preds  = [ d.predict(want, rows, k=k) for k in [1,3,5] ]
        
        add_all = True
        row_pred = {}
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
              mid1s.add( abs(want[at] - mid1[at])/sd)
              dumb.add( abs(want[at] - dumb_mid[at])/sd)

            kstat.add(   abs(want[at] - pred   )/sd)
          add_all = False
      saving_preds = False

  somes += [lrstat]
  somes += [lgbmstat]
  ## Export Predictions
  with open(f"reg/res/low-res/predictions/{dataset.split('/')[-1]}", 'w') as csv_file:  
    writer = cc.writer(csv_file)
    for key, value in predictions.items():
       k = [key]
       [k.append(v) for v in value]
       [k.append(v) for v in bl_predictions[key]]
       writer.writerow(k)  
  return somes


## Treatments: asIs, mid-leaf, 1-2-3 Nearest Neighbors, and overpopulated version of each
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


dataset = sys.argv[1]
repeats = 1
[stats.report( regression(dataset, repeats) ) ]

#d = DATA().adds(csv(dataset))
#b4 = [d.chebyshev(row) for row in d.rows]
#somes = []
#somes.append(stats.SOME(b4,f"asIs,{len(d.rows)}"))

#repeats = 1
#d = DATA().adds(csv(dataset))
#for N in (20,30,40,50):
#  the.Last = N
#  d = d.shuffle()
#  dumb = [d.clone(random.choices(d.rows, k=N)).chebyshevs().rows for _ in range(repeats)]
#  dumb = [d.chebyshev( lst[0] ) for lst in dumb]
#
#  somes.append(stats.SOME(dumb,f"dumb,{N}"))
#
#  the.Last = N
#  smart = [d.shuffle().activeLearning() for _ in range(repeats)]
#  smart = [d.chebyshev( lst[0] ) for lst in smart]
#  somes.append(stats.SOME(smart,f"smart,{N}"))
#
#stats.report(somes, 0.01)


