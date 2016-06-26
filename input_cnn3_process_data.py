import numpy as np
import cPickle as pickle
import sys

def build_data_cv(data, labels, cv=10):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    for idx in xrange(0, len(data)):
        # words = set(line[:-1].split())
        datum  = {"y":labels[idx],
                  "text": data[idx][:],
                  "num_words": data[idx][:].shape[0],
                  "split": np.random.randint(0,cv)}
        revs.append(datum)

    return revs

if __name__ == "__main__":
    print "loading data...",
    # obtain the two sentences

    first, second = [], []
    labels = []
    lines = open("first_conv-layer-output.txt").read().splitlines()
    for idx in xrange(0,len(lines)):
        first.append(lines[idx])
        first[idx] = first[idx].strip()
        first[idx] = first[idx].split("  ")
        labels.append(first[idx][-1].split()[1])
        first[idx][-1] = first[idx][-1].split()[0]
        first[idx] = [float(first[idx][i]) for i in xrange(0,len(first[idx]))]

    lines = open("second_conv-layer-output.txt").read().splitlines()
    for idx in xrange(0,len(lines)):
        second.append(lines[idx])
        second[idx] = second[idx].strip()
        second[idx] = second[idx].split("  ")
        second[idx][-1] = second[idx][-1].split()[0]
        second[idx] = [float(second[idx][i]) for i in xrange(0,len(second[idx]))]

    print "data loaded!"
    print "Concatenating..."
    input_data = np.concatenate((first,second),axis=1)
    print "Done."

    revs = build_data_cv(input_data,labels)

    pickle.dump([revs], open("input-cnnthree.p", "wb"))
    print "dataset created!"