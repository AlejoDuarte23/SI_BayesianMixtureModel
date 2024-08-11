import BSDTA as BS
import SSI_COV_AD as SSI
import Function_Footbridge as FF
import numpy as np 
import matplotlib.pyplot as plt
import MDOF_LSQ as LSQ




#.---------------------- Visual Stuff ----------------------------------------#


import itertools
from scipy import linalg
import matplotlib as mpl
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
    
    
#.---------------------- Model  ----------------------------------------#
    
    
case =2
#---------------------- 1.  Load data ----------------------------------------# 


if case == 1:
    fn = 'Data/Sp_Acc_11_02_12_7_18.txt'
    fs = 95
#---------------------- 1.  Load data ----------------------------------------# 
    fo = 16#8#16
    fi = 19#9#19
    Acc, Nc, Nd = FF.load_Data(fn,0,9,0,-1)
    fn,zeta ,phi,fopt,dampopt = SSI.SSI_COV_AD(Acc,fs,10,Nc,30,4)
    Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)
    opt,C = BS.Modal_id(Yxx,freq_id,Nc,N,fopt[-1],dampopt[-1],fo,fi)
    samples = BS.walkers(opt,N,Nc,Yxx,freq_id,6000)
    from matplotlib.colors import LogNorm
    plt.figure()
    from sklearn import mixture
    X = samples[:,:2]
    gmm = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(X)
    
    plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,'Gaussian Mixture')
    X = np.vstack((samples[:,0],samples[:,2])).T
    gmm = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(X)
    plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 1,'Gaussian Mixture')

    


    
    

if case == 2:
    fn = 'Data/DAQ_12_7_18.txt'
    fs = 1.6516*10**3
#---------------------- 1.  Load data ----------------------------------------#
    fo = 16#8#8.8#14#16
    fi = 19#10#10#16#19 
    Acc, Nc, Nd = FF.load_Data(fn,1,-1,0,-1)
    fsi = 95
    Ni = int(Acc.shape[0]*fsi/fs)
    from scipy import signal
    Acc = signal.resample(Acc, Ni)
    fn,zeta ,phi,fopt,dampopt = SSI.SSI_COV_AD(Acc,fsi,10,Nc,30,4)
    Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fsi,fo,fi)
    opt,C = BS.Modal_id(Yxx,freq_id,Nc,N,fopt[-1],dampopt[-1],fo,fi)
    samples=BS.walkers(opt,N,Nc,Yxx,freq_id,3000)
    from matplotlib.colors import LogNorm
    plt.figure()
    from sklearn import mixture
    X = samples[:,:2]
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(X)
    
    plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,'Gaussian Mixture')
    X = np.vstack((samples[:,0],samples[:,2])).T
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(X)
    plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 1,'Gaussian Mixture')

    





    
    
    
    

# plt.figure()
# LSQ.MDOF_LSQ(Acc,fs,3.5,8)

# import BSDTA as BS
# fo,fi = BS.selec_band(fopt,1)
# # fo,fi = BS.frequencie_range(fopt,dampopt,10,fs)
# i = len(fopt)-1
# plt.close('all')

# fo = 16
# fi = 20
# Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)

# # Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo[i],fi[i])
# opt,C = BS.Modal_id(Yxx,freq_id,Nc,N,fopt[-1],dampopt[-1],fo,fi)
# phi_opt,phi_pdf = BS.Opti_Modeshape(opt,freq_id,Yxx,Nc) 

# BS.ploting_results2(freq_id,Yxx,opt,0,phi_opt,Acc,fs,Nc) 


