import numpy as np 
import matplotlib.pyplot as plt
import scipy
import numdifftools as nd
import pandas as pd 
import seaborn as sb    
import emcee
import corner

from scipy.optimize import minimize
from scipy import signal
from scipy import optimize
from scipy.stats import uniform,norm

#------------------------------ 1.Theor PSD ----------------------------------#
def SDK(freq,f,z,S,Nm,N):    
    bk = f/freq
    ter1 = np.power(1-(np.power(bk, 2)),2)
    ter2 = np.power(2*z*bk,2)
    Dk = 1/(ter1+ter2)
    return S*Dk
#------------------------------ 2.likely hood --------------------------------#
def like(Y,freq_id,f,z,S,Se,phi,Nm,N,Nc): 
        Se = 10**Se
        S  = 10**S
        Li = 0
        SDKi = SDK(freq_id,f,z,S,Nm,N)
        for i in range(N):
            sk = Y[i,:].conj().reshape(Nc,1)*Y[i,:]
            ti = 2*(SDKi[i]**2+2*Se*SDKi[i]+Se**2)
            L = np.log(np.pi*ti)+(np.trace(sk)-(SDKi[i]+Se))**2/ti
            Li = Li +L
        return Li
#------------------------------ 3.Theor PSD 3 --------------------------------#   
def theor(freq,f,z,S,Se,Nm,N): 
    S = 10**S
    Se = 10**Se
    bk = f/freq
    ter1 = np.power(1-(np.power(bk, 2)),2)
    ter2 = np.power(2*z*bk,2)
    Dk = 1/(ter1+ter2)

    return S*Dk+Se
#------------------------------ 4. Get Spectral range ------------------------#  
def PSD_FORMAT(Acc,fs,fo,fi):
#    default parameters fo , fo equal to 0 and fi
    try:
        freq,Yo = signal.welch(Acc[:,0],fs,nperseg=int(len(Acc[:,0])/4)+1)
        Y = np.zeros((len(Yo),len(Acc[0,:])))
        for i in range(len(Y[0,:])):

            freq,Y[:,i] = signal.welch(Acc[:,i],fs,nperseg=int(len(Acc[:,0])/4)+1)
    except:
        freq,Y = signal.periodogram(Acc,fs)
    idd = (np.where((freq>= fo) & (freq <= fi)))
    freq_id= freq[idd]
    Yx= Y[idd]
    Yxx = Yx**0.5
    N = len(freq_id)
    
    return Yxx,freq_id,N
    

#------------------------------ 5. FDD ---------------------------------------#  
def fdd(Acc,fs,Nc):
    # Acc: Acceleration Matriz NcxN
    # fs:  Sampling Frequency
    # Nc:  Number of channels

    AN = int(len(Acc[:,0])/2)+1 # nfft/2+1
    # Memory alocation for the matrix
    PSD = np.zeros((Nc,Nc,AN),dtype=np.complex_)
    freq= np.zeros((Nc,Nc,AN),dtype=np.complex_)

    for i in range(Nc):
        for j in range(Nc):
            f, Pxy = signal.csd(Acc[:,i], Acc[:,j], fs, nfft=AN*2-1,nperseg=2**11,noverlap = None,window='hamming')
            freq[i,j]= f
            PSD[i,j]= Pxy
           
    #eigen values descomposition 
    s1 = np.zeros(len(f))
    for  i in range(len(f)):
        u, s, vh = np.linalg.svd(PSD[:,:,i], full_matrices=True)
        s1[i] = s[0]
    return s1,PSD,f

#------------------------------ 6. MacValue ---------------------------------------#  

def MacVal(Mode1,Mode2):
    ter1 =  Mode1.conj().transpose()
    ter2 =  Mode2.conj()
    num  = np.matmul(ter1, ter2)**2#np.matmul(ter1,ter2).shape
    den1  = np.matmul( Mode1.conj().transpose(), Mode1.conj())
    den2  = np.matmul( Mode2.conj().transpose(), Mode2.conj())
    den = np.matmul(den1,den2)
    Mac = num/den
    return Mac
#------------------------------ 7. ID ---------------------------------------#  

def Modal_id(Yxx,freq_id,Nc,N,fo,dampo,foo,fii):
    x = [0,0.005,-6,-20]
    phi= [0.91,0.901,0.9,0.91,0.901,0.9,0.91,0.901,0.9]
    xo = x[0:4]
    xo[0] =  freq_id[np.where(Yxx== np.max(Yxx))[0][0]]
    def posterior(x):
        
        post = like(Yxx,freq_id,x[0],x[1],x[2],x[3],phi,1,N,Nc)
             # -np.log(norm.pdf(x[0],fo,10**-2)) -np.log(norm.pdf(x[1],dampo,10**-4))
        return post
            
    likelyhood = lambda x:posterior(x)
    #---------------- Optimize likelyhood ----------------#
    opt = optimize.fmin(func=likelyhood ,x0=xo,maxiter= 1000,xtol=0.000001, ftol=0.000001)
    #---------------- Uncertainty Quanty. ----------------#
    H = nd.Hessdiag(likelyhood)(opt)
    try:
        C =np.linalg.inv(np.diag(H))
     
        suploting(opt,C)
    except:
        C = []
        print('NOT STABLE')
    return opt,C

#---------------------------- 13. Ploting Results ----------------------------#  
def ploting_results(freq_id,Yxx,opt,std,phi):

        Sxx = theor(freq_id,opt[0],opt[1],opt[2],opt[3],1,1) 
        plt.figure()
        try:
            for i in range(len(Yxx[0,:])):
                plt.plot(freq_id,Yxx[:,i],color = 'blue')
                plt.yscale('symlog')
        except:
            plt.plot(freq_id,Yxx[:,0],color = 'blue')
            plt.yscale('symlog')        
       
        plt.plot(freq_id,(Sxx*np.abs(phi[0]))**0.5,color='r')

        plt.xlabel('frequency [Hz]')
        plt.ylabel('mm/s^2/Hz^0.5')
        
        plt.show()

def ploting_results2(freq_id,Yxx,opt,std,phi,Acc,fs,Nc):

        Sxx = theor(freq_id,opt[0],opt[1],opt[2],-20,1,1) 
        plt.figure()
  
        plt.loglog(freq_id,Sxx,color='r')
        s1,PSD,f = fdd(Acc,fs,Nc)
        plt.loglog(f,s1,color='b')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('mm/s^2/Hz^0.5')
        
        plt.show()
        

#---------------------------- 14. Modeshpaes  ----------------------------#  
def ModeShapeTSBSBDA(opt,freq_id,Y,Nc):
    M = np.zeros((Y.shape[1],Y.shape[1]))
    ms = np.zeros((Y.shape[1],0))
    s1 = []
    for i in range(Y.shape[0]):
        sk = Y[i,:].conj().reshape(Nc,1)*Y[i,:]
        SDKo = SDK(freq_id,opt[0],opt[1],opt[2],1,1)
        EO =  1/(1+(opt[3]/SDKo[i]))*sk
        u, s, vh = np.linalg.svd(EO, full_matrices=True)
     
        s1.append((s[0]))
        ms =  np.column_stack((ms,np.array(u[:,0])))
        M = EO+M
    s1 = np.array(s1)
    ndices = np.where(s1 == s1.max())[0][0]
    PHI = ms[:,ndices]
    return M,PHI,s1
    
#---------------------------- 15. Likelyhood modeshapes  ----------------------#  
def Stage2VSDA(PHI,M,Se):
    ter2 = np.matmul(PHI,PHI.T)
    ter1 = np.matmul(np.matmul(PHI,M),PHI.T)
    L = -Se**-1*ter1/ter2
    return L
#---------------------------- 16. Optmal Values of Modeshapes  ---------------#  

def Opti_Modeshape(opt,freq_id,Y,Nc):
    M,PHI,s1 = ModeShapeTSBSBDA(opt,freq_id,Y,Nc)

    lik = lambda phi:Stage2VSDA(phi,M,opt[3])
    def const(phi):
        cosntt = np.linalg.norm(phi)-1
        return cosntt
    cons = {'type':'eq', 'fun': const}
    sol = minimize(lik,PHI, constraints=cons)
    
    H = nd.Hessian(lik)(sol.x)
    
    C =np.linalg.inv(H)

    Ptheta = np.random.multivariate_normal(sol.x, C, size=100)
    Ptheta2 = pd.DataFrame(Ptheta)
    plt.figure()
    pd.plotting.scatter_matrix(Ptheta2)
    name = str(opt[0])
    ext = '.png'
    plt.savefig(name+ext)
    return sol.x,Ptheta2
    

    
#---------------------------- 17. txt psd and acc record ---------------------#  
def ploting_data_file(fn,fs,fo,fi):
    Acc = np.loadtxt(fn,delimiter = ' ')
    
    plt.figure()
    plt.plot(Acc)
    plt.title('Response')
    plt.xlabel('Data points [k]')
    plt.ylabel('Accelerations [mm/s]')
    
    Yxx,freq_id,N =  PSD_FORMAT(Acc,fs,fo,fi)
    plt.figure()
    plt.plot(freq_id,10*np.log10(Yxx))
    plt.title('PSD Welch Method')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [dB]')
    
    spectrogram_plt(Acc[:,12],fs)
    
#---------------------------- 18. SPECTROGRAM --------------------------------#  

def spectrogram_plt(Acc,fs):
    plt.figure()
    cmap = plt.get_cmap('jet')
    plt.specgram(Acc, Fs =fs,cmap = cmap)
    plt.title('Spectrogram')
    plt.ylabel('frequency [Hz]')
    plt.xlabel('Time [s]')

#---------------------------- 18. Psd--------------------------------#  
def psd_scipy(Acc,fs):
    f, Pxx= signal.periodogram(Acc, fs)

#---------------------------  19. Frequency rangae ---------------------------#  
def frequencie_range(f,damp,k,fs):
    fo,fi = np.array([]),np.array([])
    for i in range(len(f)):
    
        fo = np.append(fo,np.round(f[i]*(1-k*damp[i]),fs))
        fi = np.append(fi,np.round(f[i]*(1+k*damp[i]),fs))
    return fo,fi
#---------------------------  20. Subploting  --------------------------------#  


def suploting(opt,C):
    plt.figure()
    std = np.abs(np.diag(C))**0.5
    ax1 = plt.subplot(411)
    h = np.random.normal(opt[0], scale=std[0], size=1000)
    plt.hist(h,color ='c')
    ax1.set_ylabel('f [Hz]')
    ax2 = plt.subplot(412)
    h = np.random.normal(opt[1], scale=std[1], size=1000)
    plt.hist(h,color ='g')   
    ax2.set_ylabel('zeta [k]')
    ax3 = plt.subplot(413)
    h = np.random.normal(opt[2], scale=std[2], size=1000)
    plt.hist(h,color ='b')    
    ax3.set_ylabel('S')
    ax4 = plt.subplot(414)
    h = np.random.normal(opt[3], scale=std[3], size=1000)
    ax4.set_ylabel('Se')
    try:
        plt.hist(h,color ='y')       
    except:
        s = np.random.uniform(-10,10,1000)
        plt.hist(h,color ='y')   
        plt.show()

#-------------------------- 21- Select Band ----------------------------------#
    
def selec_band(fopt,r):
    dist = np.diff(fopt)
    fo,fi = np.array([]),np.array([])
    for i in range(len(fopt)):
        if i == 0:
            fo = np.append(fo,0.001)
            fii = fopt[i]+dist[i]*r
            fi = np.append(fi,fii)
        else:
            if i ==len(dist):
                
                foo = fopt[i]-dist[i-1]*r
                fo = np.append(fo,foo)
                fi = np.append(fi,fopt[i]+5)
            else:
    
                foo = fopt[i]-dist[i-1]*r
                fii = fopt[i]+dist[i]*r
                fi = np.append(fi,fii)
                fo = np.append(fo,foo)
    return fo,fi

            
#-------------------------- 22- pymc implementation --------------------------#
def MCMCM_lIKE(Y,freq_id,f,z,S,Se,phi,Nm,N,Nc):

    def posterior(f,z,S,Se):        
        post = -like(Y,freq_id,f,z,S,Se,phi,1,N,Nc)
        return post
            
    likelyhood = lambda x:posterior(x)
    
    




#------------------------------  slampler ------------------------------------#
def walkers(xopt,N,Nc,Y,freq_id,Nsamples):

    phi= [0.91,0.901,0.9,0.91,0.901,0.9,0.91,0.901,0.9]

    std = 0.01
    
    def log_likelihood(tetha,Y,freq_id):   
        f,z,S,Se = tetha 
             
        try:
           post = -like(Y,freq_id,f,z,S,Se,phi,1,N,Nc)
        except ValueError: # NaN value case
         
           post = -np.inf # just set to negative infinity 
        return post  
        
    
    def log_prior(theta):
        f,z,S,Se = theta 
        if  0.0 < f and 0.0 < z < 1 and -100 < S < 1 and -100 < Se < -1:
            return 0.0
        return -np.inf
    
    # breakpoint()
    f =  np.random.normal(xopt[0],xopt[0]*std ,30)
    z =   np.random.normal(abs(xopt[1]),abs(xopt[1])*std,30)
    S = np.random.normal(xopt[2], -xopt[2]*std, 30)
    Se = np.random.normal(xopt[3], -xopt[3]*std,30)
    pos= np.stack((f, z,S,Se), axis=-1)
    nwalkers, ndim = pos.shape
    
    def log_probability(tetha, Y,freq_id):
        lp = log_prior(tetha)
        prob2 = lp + log_likelihood(tetha,Y,freq_id)
        if np.isnan(prob2)== 'True':
            prob2 =  -np.inf
        return prob2
    

    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,args=(Y,freq_id))

    sampler.run_mcmc(pos, Nsamples, progress=True)
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=100,flat=True)
    labels = ["f", "z", "log10(S)", "log10(Se)"]
    fig = corner.corner(flat_samples, labels=labels);
    return flat_samples
    
    

        
    