import numpy as np
import matplotlib.pyplot as plt

#---------------------- 1.  Load data ----------------------------------------# 

def load_Data(fn,cho,chi,i,j):
    Acco = np.loadtxt(fn)
    Acc = Acco[i:j,cho:chi]
    Nc = Acc.shape[1]
    Nd = Acc.shape[0]
    return Acc, Nc, Nd


