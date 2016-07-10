def store_sent(batches, num, datasets):
    p_sento_finale = []
    sento_finale = []
    print "sentence {0} batch len {1}".format(num, len(batches))
    for batch in xrange(0, len(batches)):
        for sentence_id in xrange(0,len(batches[batch])):
            if len(datasets[0]) > sentence_id:
                off = []
                p_off = []
                off = np.append(batches[batch][sentence_id],datasets[0][sentence_id,-1])
                p_off = np.append(batches[batch][sentence_id],(datasets[0][sentence_id,-1]))
                sento_finale.append(off)
                p_sento_finale.append(p_off)

    print "sentences in {0} concatenated. {1}".format(num, len(first_sent[0]))
    sys.stdout.flush()

    return sento_finale, p_sento_finale

def file_save(sentences, file_name):
    file = file_name+".txt"
    f = open(file,"w")
    print "Saving into text files"
    sys.stdout.flush()
    for sent in xrange(0, len(sentences)):
        for br in xrange(0,len(sentences[sent])):
            if (br+1)==len(sentences[sent]):
                f.write('%d' % sentences[sent][br])
            else:
                f.write('%10.6f ' % sentences[sent][br])

        f.write("\n")
    f.close()

def store_output(first_sent, second_sent, datasets):
    s1, p1 = store_sent(first_sent, 0, datasets)
    s2, p2 = store_sent(second_sent, 1, datasets)
    file_save(s1, "first_conv-layer-output")
    file_save(p1, "first_conv-layer-output-prob")
    file_save(s1, "second_conv-layer-output")
    file_save(p1, "second_conv-layer-output-prob")