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

#def build_me(data, f_p_y_given_xs1,f_weights11,f_weights21,f_weights31,f_weights41,f_bias11,f_bias21,f_bias31,f_bias41,f_p_y_given_xs2,f_weights12,f_weights22,f_weights32,f_weights42,f_bias12,f_bias22,f_bias32,f_bias42):
def build_me(f_weights21,f_weights31,f_weights41,f_bias11,f_weights12,f_weights22,f_weights32,f_weights42,f_bias12):
    # print "loading data...",
    # revs = build_data_cv(data, cv=10)
    # max_l = np.max(pd.DataFrame(revs)["num_words"])
    # print "data loaded!"
    # print "number of sentences: " + str(len(revs))
    # print "max sentence length: " + str(max_l)
    #cPickle.dump([revs, f_p_y_given_xs1,f_weights11,f_weights21,f_weights31,f_weights41,f_bias11,f_bias21,f_bias31,f_bias41,f_p_y_given_xs2,f_weights12,f_weights22,f_weights32,f_weights42,f_bias12,f_bias22,f_bias32,f_bias42], open("new_dump.p", "wb"))
    print "Pickling..."
    cPickle.dump([f_p_y_given_xs1,f_p_y_given_xs2,f_weights11,f_weights21,f_weights31,f_weights41,f_bias11,f_weights12,f_weights22,f_weights32,f_weights42,f_bias12], open("new_dump.p", "wb"))
    print "dataset created!"
