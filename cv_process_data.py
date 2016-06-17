import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

def build_data_cv(data, cv=10):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    for line in data:
        # words = set(line[:-1].split())
        datum  = {"y":line[-1], 
                  "text": line[:-1],                             
                  "num_words": line[:-1].shape[0],
                  "split": np.random.randint(0,cv)}
        revs.append(datum)
    
    return revs

def build_me(data, W):   
    print "loading data...",        
    revs = build_data_cv(data, cv=10)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "max sentence length: " + str(max_l)
    cPickle.dump([revs, W], open("new_dump.p", "wb"))
    print "dataset created!"