import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

def process(idx, rev, label, file_type, vocab, clean_string=True):
    """
    Defines a label, text, num_words and type for each sentence
    label in [0,1,2]
    text - the sentence under consideration
    num_words - count of words present in text
    type in [train, valid, test]
    """
    # rev=[]
    if clean_string:
        orig_rev = clean_str(" ".join(rev))
    else:
        orig_rev = " ".join(rev).lower()

    words = set(orig_rev.split())
    for word in words:
        vocab[word] += 1

    datum  = {"label":label, 
              "text": orig_rev,                             
              "num_words": len(orig_rev.split()),
              "type":file_type,
              "idx":idx}

    return datum, vocab

def add_logic(file, file_type, revs, vocab, split_sent, clean_string=True):
    """
    Deals with the logic behind a saved sentence-label relationship
    Supports split sentence in twos
    """
    with open(file, "rb") as f:
        for line in f:
            # sentence = []
            # label = []
            data = line.split("\t")
            label = data[0]
            label = label.strip()
            sentence = data[1] 
            sentence = sentence.strip()
            if split_sent:
                sentence = sentence.split(".")
                # get rid of potential empty spaces
                sentence[0] = sentence[0].strip()
                sentence[1] = sentence[1].strip()
            
            if type(sentence) == list:
                for idx in xrange(2):
                    rev = []
                    rev.append(sentence[idx])
                    datum, vocab = process(idx ,rev, label, file_type, vocab, clean_string)
                    revs.append(datum)
            else:
                rev = []
                rev.append(sentence)
                datum, vocab = process(sentence, 0,rev, label, file_type, vocab, clean_string)
                revs.append(datum)

    return revs,vocab


def build_data(data_folder, split_sent, clean_string=True):
    """
    Loads data and adds logic.
    """
    revs = []
    train_file = data_folder[0]
    valid_file = data_folder[1]
    test_file = data_folder[2]
    vocab = defaultdict(float)
    # dedicate_logic

    for idx in xrange(len(data_folder)):
        # 0 train, 1 valid, 2 test
        # chained ternary operator
        data_type = "train" if idx==0 else "valid" if idx==1 else "test"
        # update revs and vocab
        revs, vocab = add_logic(data_folder[idx], data_type, revs, vocab, split_sent)

    return revs, vocab
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_glove_vec(fname, vocab):
  """
  Loads 300x1 word vecs from Stanford (Socher) GloVe
  """
  word_vecs = {}
  target = open(fname, "r")

  word_vecs = {}
  for item in target:
    elements = item.split()
    word = elements[0]
    if word in vocab:
      word_vecs[word] = np.asarray(elements[1:], dtype='float32')

  target.close()
  return word_vecs

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\.{2}", ".", string)
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":    
    w2v_file = sys.argv[1]
    split_sent = sys.argv[2]
    if split_sent=="False":
        split_sent = False
    data_folder = ["train.txt","valid.txt","test.txt"]   
    print "loading data...",        
    revs, vocab = build_data(data_folder, split_sent, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    who = "word2vec"
    if "glove" in w2v_file:
        who = "GloVe"
        print "loading GloVe vectors...",
    else:   
        print "loading {0} vectors...".format(who),
    w2v = load_glove_vec(w2v_file, vocab)
    print "{0} loaded!".format(who)
    print "num words already in {0}: {1}".format(who, str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("snli-GloVe-Split.p", "wb"))
    print "dataset created!"
    
