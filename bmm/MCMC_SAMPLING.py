import corner
import numpy as np
import matplotlib.pyplot as plt
import emcee


#------------------------------  slampler ------------------------------------#
def walkers(xopt):
    pos = xopt + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape
    return pos,nwalkers,ndim
            

    
def sample_MCMC(pos,nwalkers, ndim, log_probability, argso,Nsamples):
    
# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args= argso)

    sampler.run_mcmc(pos, Nsamples, progress=True)
    samples = sampler.get_chain()
    
    return samples,sampler

def plotresuls(sampler,labels):
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    print(flat_samples.shape)
    fig = corner.corner(flat_samples, labels=labels);
    
    


