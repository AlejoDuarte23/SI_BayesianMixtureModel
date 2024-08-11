import scipy
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
import scipy.stats as stats

from scipy.optimize import least_squares
from scipy import signal
from tqdm import tqdm

Nm = 5
#--------------------------- 0. Turn off warnings -----------------------------#
if not sys.warnoptions:
    warnings.simplefilter("ignore")
plt.close('all')  
# Theor PSD 
def H(freq,f,z,S,Se,Nm,N):
    H = np.zeros((Nm,Nm,N),dtype=np.complex_)
    for i in range(0,Nm):
        for j in range(0,Nm): 
            if i==j:
                bki = f[i]/freq
                bkj = f[j]/freq
                ter1 = 1/((1-bki)+2j*z[i]*bki)
                ter2 = 1/((1-bkj)+2j*z[j]*bkj)
                H[i,j,:] = (10**S[i,j])*(ter1)*(ter2)+10**Se
    return H
#--------------------------- 1. Likely ---------------------------------------#
def likelihood(x,freq,ymed,Nm,N):
  
    f = [x[0],x[1],x[2],x[3],x[4]]
    z = [x[5],x[6],x[7],x[8],x[9]]
    S= np.zeros((Nm,Nm))
    S[0,0] = x[10]
    S[1,1] = x[11]
    S[2,2] = x[12]
    S[3,3] = x[13]
    S[4,4] = x[14]
    
    Se=x[15]
    modelo = np.array([])

    H1 = H(freq,f,z,S,Se,Nm,N)
    for i in range(len(freq)):
     
        ESY = H1[:,:,i]+10**-40
        modelo = np.abs(np.append(modelo,np.trace(ESY)/N))
    return modelo-ymed

#--------------------------- 2. Plot PSD --------------------------------------#
def plot_psd(x,Nm,N,freq_id,s1_id):
    f = [x[0],x[1],x[2],x[3],x[4]]
    z = [x[5],x[6],x[7],x[8],x[9]]
    S= np.zeros((Nm,Nm))
    S[0,0] = x[10]
    S[1,1] = x[11]
    S[2,2] = x[12]
    S[3,3] = x[13]
    S[4,4] = x[14]
    Se = x[15]
    H1 = H(freq_id,f,z,S,Se,Nm,N)
    plt.figure(1)
    for i in range(Nm):
        plt.plot(freq_id,10*np.log10(np.trace(H1)/N),'r',label = 'E[Sy]')
    plt.plot(freq_id,10*np.log10(s1_id),'b',label ='Single Value Spectrum')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Decibel [Db]')


#--------------------------- 5. fdd ------------------------------------------#


def fdd(Acc,fs,Nc):
    # Acc: Acceleration Matriz NcxN
    # fs:  Sampling Frequency
    # Nc:  Number of channels

    AN = 2049 # nfft/2+1
    # Memory alocation for the matrix
    PSD = np.zeros((Nc,Nc,AN),dtype=np.complex_)
    freq= np.zeros((Nc,Nc,AN),dtype=np.complex_)

    for i in range(Nc):
        for j in range(Nc):
            f, Pxy = signal.csd(Acc[:,i], Acc[:,j], fs, nfft=4096,nperseg=2**11,noverlap = None,window='hamming')
            freq[i,j]= f
            PSD[i,j]= Pxy
           
    #eigen values descomposition 
    s1 = np.zeros(len(f))
    for  i in range(len(f)):
        u, s, vh = np.linalg.svd(PSD[:,:,i], full_matrices=True)
        s1[i] = s[0]
    return s1,PSD,f

#--------------------------- 4. Load txt -------------------------------------#
    
def load_data(name,delim,fs,cho,ch1):
    A = np.loadtxt(name,delimiter=delim)
    ACC = A[:,cho:ch1]
    return ACC
#--------------------------- 5. Load txt -------------------------------------#
def MDOF_LSQ(ACC,fs,fo,fi):
    #Frequency domain descomposition
    s1,psd,freq = fdd(ACC,fs,len(ACC[1,:]))
    # INitial p
    # Selecting frequency band
    idd = (np.where((freq>= fo) & (freq <= fi)))
    freq_id= freq[idd]
    s1_id= s1[idd]
    N = len(freq_id)
    # Single valur spectrum
    plt.figure(3)
    plt.plot(freq_id,10*np.log10(s1_id))
  
    
    # fo = [65.5,70.83,80.12,87.27,100]
    # z = [0.001,0.001,0.001,0.001,0.001]
    # S = [-13,-12,-11,-12,-13]
    # Se = [-10]
    xo = [3.7,3.9,4.8,5.6,7,0.01,0.01,0.01,0.01,0.01,\
          -3,-1,-1,-3,-3,1]
    
    likelyhood = lambda xo,freq,si: likelihood(xo,freq,si,Nm,N)
    opt = least_squares(likelyhood ,xo,loss='cauchy',f_scale=0.1,args=(freq_id, s1_id))
    plot_psd(opt.x,Nm,N,freq_id,s1_id)
