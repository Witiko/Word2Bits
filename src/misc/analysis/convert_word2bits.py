# Ttakes w2v bin format and converts it to glove format.
# Usage: python convert_word2bits.py input_binary_wordvecs output_file [threshold_value]
from __future__ import print_function
import sys
import random
import math
import numpy as np

threshold = 0

def threshold_value(x):
    if threshold == 0:
        return x
    retval = 0
    sign = -1 if x < 0 else 1
    x *= sign    
    if threshold == 1:
        return sign / 3
    elif threshold == 2:
        if x >= 0 and x <= .5:
            retval = .25
        else:
            retval = .75
    else:
        assert(0)
    return sign * retval

def load_vec(fname):
    word_vecs = {}
    print("Loading vectors from %s" % fname)
    with open(fname, "r") as f:
        n_words, dimension = [int(x) for x in f.readline().split(" ")]
        print(n_words, dimension)
        for i in range(n_words):
            if i % 10000 == 0:
                print("%d of %d" % (i, n_words))
            line = f.readline().split(" ")
            word, vector = line[0], [float(x) for x in line[1:]]
            word_vecs[word] = vector
        assert len(word_vecs.keys()) == n_words
    return word_vecs
    
def load_bin_vec(fname):
    word_vecs = {}
    ordering = []
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        print(vocab_size, layer1_size)
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
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
        return word_vecs

def write_bin_vec_mikolov(vecs, fname_out):
    with open(fname_out, "wb") as f:
        vocab_size, layer1_size = len(vecs.items()), len(vecs.values()[0])
        print("Writing: %d %d" % (vocab_size, layer1_size));
        print("%d %d" % (vocab_size, layer1_size), file=f)
        binary_len = np.dtype('float32').itemsize * layer1_size
        for word, vector in vecs.items():
            print("%s " % word, file=f, end='')
            vector.astype('float32').tofile(f)
            print("", file=f)

def write_bin_vec_compressed(vecs, fname_out):
    with open(fname_out, "wb") as f:
        vocab_size, layer1_size = len(vecs.items()), len(vecs.values()[0])
        unique_values = np.unique(np.vstack(vecs.values()))
        n_bits = int(np.ceil(np.log(len(unique_values))))
        assert 2**n_bits >= len(unique_values)
        assert 8 % n_bits == 0
        print("%d %d %d" % (vocab_size, layer1_size, n_bits), file=f)
        n_codes_per_byte = 8 / n_bits
        layer1_size_rounded = int(math.ceil(layer1_size / float(n_codes_per_byte)) * n_codes_per_byte)
        value_to_code = {value : i for i, value in enumerate(unique_values)}
        sorted_values = [x[0] for x in sorted(value_to_code.items(), key=lambda x:x[1])]
        print("\n".join([str(x) for x in sorted_values]), file=f)
        for word, vec in vecs.items():
            print("%s " % word, end='', file=f)
            assert(len(vec) == layer1_size)
            assert(len(vec) <= layer1_size_rounded)
            for i in range(0, layer1_size_rounded, n_codes_per_byte):
                code = 0
                for j in range(n_codes_per_byte):
                    if i+j >= layer1_size:
                        break
                    code <<= n_bits
                    code |= value_to_code[vec[i+j]]
                assert code < 256 and code >= 0
                f.write(chr(code))
            print("", file=f)

def write_bin_vec_text(vecs, fname_out):
    with open(fname_out, "w") as f:
        vocab_size, layer1_size = len(vecs.items()), len(vecs.values()[0])        
        print("%d %d" % (vocab_size, layer1_size), file=f)        
        for name, vec in vecs.items():
            vec_string = " ".join([str(x) for x in vec])
            print("%s %s" % (name, vec_string), file=f)
            
def count(x):
    return np.sum(x == .5)


if __name__=="__main__":
    
    assert len(sys.argv) >= 3
    
    threshold = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    word_vecs = load_bin_vec(sys.argv[1])
    
    if threshold != 0:
        print("Thresholding: %d" % threshold)
        threshold_function = np.vectorize(threshold_value)
        for i, (k,vec) in enumerate(word_vecs.items()):
            if i % 100000 == 0:
                print("%d of %d" % (i, len(word_vecs)))
            word_vecs[k] = threshold_function(vec)
            
    write_bin_vec_text(word_vecs, sys.argv[2])
