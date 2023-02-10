import numpy as np

alphabet = "XACDEFGHIKLMNOPQRSTUVWY"
c2i_dict = dict((c,i) for i, c in enumerate(alphabet))
i2c_dict = dict((i,c) for i, c in enumerate(alphabet))

def encode_sample(sample):
    sample = sample.replace("B","N")
    sample = sample.replace("Z","Q")
    sample = sample.replace("J","L")
    n_arr = [c2i_dict[a] for a in sample]
    return np.array(n_arr)

def findOccurrences(seq, symbol):
    return [i for i, letter in enumerate(seq) if letter == symbol]

def getSamplewithLength(seq,idx,sample_length=20):
    n_str = f"{'X'*20+seq+'X'*20}"
    n_str = "".join(n_str.splitlines())
    idx=idx+20
    l_idx= (idx-sample_length)
    h_idx=  (idx+sample_length+1)
    n_str = n_str[int(l_idx):int(h_idx)]
    assert( len(n_str)==41)
    return n_str

def samplesfromProtein(symbol,seq):
    ## creating a set of all occurrences of a symbol
    symbol_idx_occur_list = findOccurrences(seq, symbol)
    sample_list= list()
    for idx in symbol_idx_occur_list:
        tmp = getSamplewithLength(seq,int(idx))
        assert(len(tmp)==41)
        #sample_list.append(getSamplewithLength(seq,int(idx)))
        sample_list.append(tmp)
    return sample_list, symbol_idx_occur_list

