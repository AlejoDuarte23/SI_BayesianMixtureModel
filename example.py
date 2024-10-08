import bmm as BS
import numpy as np 
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl

from matplotlib.colors import LogNorm
from sklearn import mixture
from scipy import linalg
from sklearn import mixture
from scipy import signal

mpl.use('Qt5Agg')
mpl.pyplot.ion()


#%% Util functions
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    

def load_Data(fn,cho,chi,i,j):
    Acco = np.loadtxt(fn, delimiter=',')
    Acc = Acco[i:j,cho:chi]
    Nc = Acc.shape[1]
    Nd = Acc.shape[0]
    return Acc, Nc, Nd
 
#%%    
match 'mixture':
    case 'single':
        fn = 'data/Sp_Acc_11_02_12_7_18.txt'
        fs = 95
    #---------------------- 1.  Load data ----------------------------------------# 
        fo = 8#16
        fi = 9#19
        Acc, Nc, Nd = load_Data(fn,0,9,0,-1)
        Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)
        opt,C = BS.Modal_id(Yxx,freq_id,Nc,N,8.72,0.02,fo,fi)
        samples = BS.walkers(opt,N,Nc,Yxx,freq_id,6000)
        
   

    case 'mixture':
        fn = 'data/DAQ_12_7_18.txt'
        fs = 1.6516*10**3
    #---------------------- 1.  Load data ----------------------------------------#
        fo = 16#8#8.8#14#16
        fi = 19#10#10#16#19 
        Acc, Nc, Nd = load_Data(fn,1,-1,0,-1)
        fsi = 95
        Ni = int(Acc.shape[0]*fsi/fs)
        Acc = signal.resample(Acc, Ni)
        Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fsi,fo,fi)
        opt,C = BS.Modal_id(Yxx,freq_id,Nc,N,18.3,0.05,fo,fi)
        samples=BS.walkers(opt,N,Nc,Yxx,freq_id,6000)
        plt.figure()
        X = samples[:,:2]
        gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(X)
        
        plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,'Gaussian Mixture')
        X = np.vstack((samples[:,0],samples[:,2])).T
        gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(X)
        plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 1,'Gaussian Mixture')
