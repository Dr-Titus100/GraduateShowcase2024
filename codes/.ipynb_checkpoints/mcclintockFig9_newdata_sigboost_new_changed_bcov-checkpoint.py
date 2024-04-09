#--------------- Packages ---------------#
import os
import sys
import emcee
import fitsio
import Corrfunc
import numpy as np
from classy import Class
import cluster_toolkit as ctk
from getdist import plots, MCSamples
from colossus.halo import concentration
from colossus.cosmology import cosmology
from scipy.stats import norm#, multivariate_normal

from read_mini_uchuu import ReadMiniUchuu
from measure_lensing_v2 import MeasureLensing
import numdifftools as nd
import argparse
from schwimmbad import MPIPool 
from sklearn.linear_model import LinearRegression

###################################################################
def log_prior_individual(param, mu, sigma):
    return -0.5*((param - mu)/sigma)**2

def log_gaussian_prior(theta):
    tau, fmis, Am = theta
    return log_prior_individual(tau, tau_mu_prior, tau_sigma_prior) + \
           log_prior_individual(fmis, fmis_mu_prior, fmis_sigma_prior) + \
           log_prior_individual(Am, Am_mu_prior, Am_sigma_prior)

def log_flat_prior(args):
    log10_M, c, B0, Rs = args
    if 11.0<log10_M<18.0 and 0.0<c<20.0 and 0.0<B0<10.0 and 0.0<Rs<10.0: #change from mvir to log_mvir
        return 0.0
    return -np.inf

def log_probability(params, data, sac, boost_data, z, lam, boost_cov):
    lp = log_flat_prior(params[:4]) + log_gaussian_prior(params[4:])
    if not np.isfinite(lp):
        return -np.inf
    lp2 = lp + log_likelihood(params, data, sac, boost_data, z, lam, boost_cov)[0]
    if np.isnan(lp2):
        return -np.inf
    else: 
        return lp2

def log_likelihood(params, DS_data, sac, boost_data, z, lam, boost_cov):
    # log10_M, c = params
    log10_M, c, B0, Rs, tau, fmis, Am = params
    # sigRs, sigB0 = sigboost_params
    # log10_M = log10_M + np.log10(h) #McClintock units have no h.
    M = 10**log10_M #Msun/h
    # z = 0.6

    # computing miscentering corrections
    Rlam = (lam/100)**0.2 #Mpc/h comoving #cluster radius assigned by redmapper
    Rmis = tau*Rlam #Mpc/h Radial miscentering offset. Cluster centers are wrongly identified by a distance Rmis.
    Rproj = np.logspace(-2, 2.4, num=1000, base=10) #Mpc/h Projected 2D radii. 
    R3d = np.logspace(-2, 3, num=1000, base=10) #Mpc/h comoving. 3D radii.
    
    # McClintock radial bins
    Rmin = 0.0323
    Rmax = 30
    nbins = 15
    
    Redges = np.logspace(np.log10(Rmin), np.log10(Rmax), nbins+1) #Projected radial bin edges
    # Redges = np.logspace(np.log10(0.0323), np.log10(30.), num=15+1) #Projected radial bin edges
    Redges *= h*(1+z) #Converted to Mpc/h comoving

    """
    Note:
    Minimum Rproj for Sigma(Rproj) must be >= than min(r) of xi(r).
    Maximum Rproj for Sigma(Rproj) must be <= than max(r) of xi(r).
    Thus, the range of values for Rproj must be 
    equal to (or contained in) that of r
    """

    # Compute boost factors from cluster toolkit
    #Note: Rs is default in Mpc physical
    boost_model = ctk.boostfactors.boost_nfw_at_R(Rproj, B0, Rs*h*(1+z)) #theory
    
    #Specify k and z
    # k = np.logspace(-5, 3, num=4000) #Mpc^-1 comoving
    k = np.logspace(-5, np.log10(k_max), num=4000) #Mpc^-1 comoving
    # Power spectrum
    Pnonlin = np.array([cosmo_ctk.pk(ki, z) for ki in k])#*h**3  #Mpc^3/h^3 comoving
    Plin = np.array([cosmo_ctk.pk_lin(ki, z) for ki in k])#*h**3  #Mpc^3/h^3 comoving
    kh = k/h #h/Mpc comoving
    #P(k) are in Mpc^3/h^3 comoving
    #Thus, you will need to convert these to h/Mpc and (Mpc/h)^3 to use in the toolkit.
    Plin *= h**3
    Pnonlin *= h**3

    # NFW profile
    xi_nfw = ctk.xi.xi_nfw_at_r(R3d, M, c, Omega_m)

    # # Matter-matter correlation function (matter auto-correlation)
    xi_mm = ctk.xi.xi_mm_at_r(R3d, kh, Pnonlin)

    # 2-halo correlation function
    bias_term = ctk.bias.bias_at_M(M, kh, Plin, Omega_m) # Here, P must be linear.
    xi_2halo = ctk.xi.xi_2halo(bias_term, xi_mm)

    # Halo-matter correlation function
    xi_hm = ctk.xi.xi_hm(xi_nfw, xi_2halo)

    # Sigma (computed from xi_hm)
    Sigma = ctk.deltasigma.Sigma_at_R(Rproj, R3d, xi_hm, M, c, Omega_m) #Sigma

    # DeltaSigma (excess surface density)
    DS = ctk.deltasigma.DeltaSigma_at_R(Rproj, Rproj, Sigma, M, c, Omega_m) #DeltaSigma

    Sigma_mis  = ctk.miscentering.Sigma_mis_at_R(Rproj, Rproj, Sigma, M, c, Omega_m, Rmis, kernel="gamma") #miscentered Sigma profiles
    DS_mis = ctk.miscentering.DeltaSigma_mis_at_R(Rproj, Rproj, Sigma_mis) #miscentered Sigma profiles

    #full Sigma profile; i.e. miscentered + correctly centered Sigma profiles
    full_Sigma = (1-fmis)*Sigma + fmis*Sigma_mis 
    #full DeltaSigma profile; i.e. miscentered + correctly centered DeltaSigma profiles
    full_DS = (1-fmis)*DS + fmis*DS_mis 
    full_DS *= Am #multiplicative bias due to shear and photometric redshift.

    full_DS /= boost_model #de-boost the model
    Sigma_crit_inv = Sigma_crit_inv0*h*(1+z)**2
    full_DS /= (1-full_Sigma*Sigma_crit_inv) #Reduced shear
    #Here, DeltaSigma is in Msun h/pc^2 comoving

    ave_DS = ctk.averaging.average_profile_in_bins(Redges, Rproj, full_DS)
    ave_DS *= h*(1+z)**2 #convert to Msun/pc^2 physical
    
    # DS radii cut
    inds = (rp > 0.1)*(rp < 999) #rp is the radial bins for DS. It's a global variable. M19 cuts at 0.2.
    cov_cut = sac[inds]
    cov_cut = cov_cut[:,inds]
    icov_cut = np.linalg.inv(cov_cut)
    icov_all = np.linalg.inv(sac)
    DS_model = ave_DS[inds]
    Bp1 = boost_data#[threshold_cut]
    Bp1_cut = Bp1[inds]

    #------------------Boost factor covariance
    iBcov_cut = boost_cov[inds]
    iBcov_cut = iBcov_cut[:,inds]
    
    #------------------Boost factor model
    # Here, McClintock did not convert Rs to Rs*h*(1+z)
    boost_model2 = ctk.boostfactors.boost_nfw_at_R(rp, B0, Rs) #compares with data
    
    # Difference between data and model and the likelihood.
    data_diff = DS_data[inds] - DS_model
    boost_diff = Bp1_cut - boost_model2[inds]
    data_likelihood = -0.5*np.dot(data_diff, np.dot(icov_cut, data_diff))
    boost_likelihood = -0.5*np.dot(boost_diff, np.dot(iBcov_cut, boost_diff))
    total_likelihood =  data_likelihood + boost_likelihood
    return total_likelihood, DS_model#, ave_DS

def add_sys_to_data(mass, DS_data0, Sigma_data0, Rs, B0, Am, z, lam, Rp_mid):
    tau = 0.17
    fmis = 0.25
    # mass = 10**log10_M
    c = concentration.concentration(M=mass, mdef='200m', z=z, model = 'bhattacharya13') # 'bhattacharya13', 'bullock01'
    # computing miscentering corrections
    Rlam = (lam/100)**0.2 #Mpc/h comoving
    Rmis = tau*Rlam #Mpc/h Radial miscentering offset. Cluster centers are wrongly identified by a distance Rmis.
    
    # McClintock radial bins
    Rmin = 0.0323
    Rmax = 30
    nbins = 15
    Rproj = rp_
    Redges = np.logspace(np.log10(Rmin), np.log10(Rmax), nbins+1) #Projected radial bin edges
    Redges *= h*(1+z) #Converted to Mpc/h comoving

    Sigma_mis  = ctk.miscentering.Sigma_mis_at_R(Rproj, Rproj, Sigma_data0, mass, c, Omega_m, Rmis, kernel="gamma") #miscentered Sigma profiles
    DS_mis = ctk.miscentering.DeltaSigma_mis_at_R(Rproj, Rproj, Sigma_mis) #miscentered Sigma profiles

    full_Sigma = (1-fmis)*Sigma_data0 + fmis*Sigma_mis 
    DS_data = (1-fmis)*DS_data0 + fmis*DS_mis # miscentered (from theory) + correctly centered DeltaSigma profiles
    DS_data *= Am # multiplicative bias

    # Compute boost factors from cluster toolkit
    boost_model = ctk.boostfactors.boost_nfw_at_R(Rproj, B0, Rs*h*(1+z)) # to correct for dilution effects
    DS_data /= boost_model # de-boost the model
    Sigma_crit_inv = Sigma_crit_inv0*h*(1+z)**2
    DS_data /= (1-full_Sigma*Sigma_crit_inv) #Reduced shear
    DS_data_final = ctk.averaging.average_profile_in_bins(Redges, Rproj, DS_data)
    DS_data_final *= h*(1+z)**2 #convert to Msun/pc^2 physical
    boost_model2 = ctk.boostfactors.boost_nfw_at_R(Rp_mid, B0, Rs) 
    return DS_data_final, boost_model2#, Rp_mid

def run_mcmc(data, params, nwalkers, nsteps, burnin, sac, boost_data, z, readerfile, lam, boost_cov): 
    init_pts = params
    ndim = len(init_pts) #number of params we want to calibrate

    # MPI parallelization
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        
        # # Set up the backend
        if not os.path.isfile(readerfile):
            backend = emcee.backends.HDFBackend(readerfile)
            backend.reset(nwalkers, ndim)
            pos = [init_pts + 1e-3*np.random.randn(ndim) for k in range(nwalkers)]
        else:
            backend = emcee.backends.HDFBackend(readerfile)
            pos = None

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(np.array(data), sac, boost_data, z, lam, boost_cov), 
                                        backend=backend)
        sampler.run_mcmc(pos, nsteps, progress=True);

    af = sampler.acceptance_fraction
    print("Mean acceptance fraction:", np.mean(af))

    samples_n = sampler.flatchain#.copy()
    O_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                 zip(*np.percentile(samples_n, [16, 50, 84], 
                                    axis=0)))
    print('\nResults from emcee')
    print(O_mcmc)
    flat_samples = sampler.get_chain(discard=burnin, flat=True)
    return sampler, flat_samples, samples_n, O_mcmc

######################################################
## Richness-mass relation
def evrard_extra_term(richness, mass, C, random_state):#, x_cod, y_cod, z_cod):
    all_mass = norm.rvs(mass, 0.25, random_state=random_state)#mass
    ln_lam0 = richness #norm.rvs(richness, 0.25) #richness
    selection = np.exp(ln_lam0) >= 20
    ln_lam = ln_lam0[selection]
    ln_mass = all_mass[selection] # dependent variable Y
    
    # linear regression model
    lm = LinearRegression() # Construct model
    lm.fit(ln_lam.reshape(-1, 1), ln_mass)
    
    # model parameters
    intercept = lm.intercept_ # the intercept (B).
    slope = lm.coef_ # the slope (A).
    rsquared = lm.score(ln_lam.reshape(-1, 1), ln_mass) # the r-squared value
    reg_line = lm.predict(ln_lam.reshape(-1, 1))
    inv_reg_line = (ln_mass - intercept)/slope
    
    # computing alpha and pi
    alpha = 1/slope
    pi = -alpha*(intercept + C)
    C = - (pi/alpha + intercept)
    log_richness0 = alpha*all_mass + pi #no scatter
    return log_richness0, alpha, pi, C, rsquared, ln_mass, all_mass#, x_cod_new, y_cod_new, z_cod_new
    
def evrard_extra_term2(richness, mass, C):#, x_cod, y_cod, z_cod):
    all_mass = mass
    # selection = np.exp(richness) >= 20
    ln_lam0 = richness #norm.rvs(richness, 0.25) #richness
    selection = np.exp(ln_lam0) >= 20
    ln_lam = ln_lam0[selection]
    ln_mass = all_mass[selection] # dependent variable Y
    
    # linear regression model
    lm = LinearRegression() # Construct model
    lm.fit(ln_lam.reshape(-1, 1), ln_mass)
    
    # model parameters
    intercept = lm.intercept_ # the intercept (B).
    slope = lm.coef_ # the slope (A).
    rsquared = lm.score(ln_lam.reshape(-1, 1), ln_mass) # the r-squared value
    reg_line = lm.predict(ln_lam.reshape(-1, 1))
    inv_reg_line = (ln_mass - intercept)/slope
    
    # computing alpha and pi
    alpha = 1/slope
    pi = -alpha*(intercept + C)
    C = - (pi/alpha + intercept)
    log_richness0 = alpha*all_mass + pi #no scatter
    
    return log_richness0, alpha, pi, C, rsquared, ln_mass#, x_cod_new, y_cod_new, z_cod_new
################################################################

################################################################
def getargs(lam_bin, z_bin):
    # Values taken from McClintock et. al., 2019.
    lam_z_bin = "l"+str(lam_bin)+"_z"+str(z_bin)
    
    # Masses
    mass_dic = {"l3_z0": mean_mass_true0, "l4_z0": mean_mass_true1, 
                "l5_z0": mean_mass_true2, "l6_z0": mean_mass_true3}
    
    # DeltaSigma profiles
    profile_dic = {"l3_z0": DeltaSigma0, "l4_z0": DeltaSigma1, "l5_z0": DeltaSigma2, "l6_z0": DeltaSigma3} 
    Sigma_dic = {"l3_z0": Sigma0, "l4_z0": Sigma1, "l5_z0": Sigma2, "l6_z0": Sigma3} 
    
    # Concentrations
    c_dic = {"l3_z0":5.81 , "l4_z0":4.53, "l5_z0":4.38, "l6_z0":4.65, 
             "l3_z1":5.68 , "l4_z1":6.24, "l5_z1":5.41, "l6_z1":3.19, 
             "l3_z2":4.76, "l4_z2":3.61, "l5_z2":4.76, "l6_z2":3.73}

    B_dic = {"l3_z0":0.34, "l4_z0":0.37, "l5_z0":0.27, "l6_z0":0.23, 
         "l3_z1":0.05, "l4_z1":0.14, "l5_z1":0.05, "l6_z1":0.21, 
         "l3_z2":0.13, "l4_z2":0.13, "l5_z2":0.09, "l6_z2":0.04}

    sigB_dic = {"l3_z0":0.05, "l4_z0":0.06, "l5_z0":0.05, "l6_z0":0.03, 
              "l3_z1":0.01, "l4_z1":0.04, "l5_z1":0.02, "l6_z1":0.17, 
              "l3_z2":0.05, "l4_z2":0.08, "l5_z2":0.06, "l6_z2":0.04}

    # Boost factor scale radius
    Rs_dic = {"l3_z0":0.44, "l4_z0":0.50, "l5_z0":0.80, "l6_z0":1.37, 
              "l3_z1":0.89, "l4_z1":0.44, "l5_z1":1.72, "l6_z1":0.51, 
              "l3_z2":0.38, "l4_z2":0.44, "l5_z2":0.85, "l6_z2":35.94}

    sigRs_dic = {"l3_z0":0.06, "l4_z0":0.07, "l5_z0":0.15, "l6_z0":0.21, 
              "l3_z1":0.24, "l4_z1":0.10, "l5_z1":0.95, "l6_z1":0.23, 
              "l3_z2":0.11, "l4_z2":0.18, "l5_z2":0.37, "l6_z2":29.69}
    
    # redshift
    z_dic = {"0":0.3, "1":0.45, "2":0.6}
    # z_dic = {"0":0.5*(0.2+0.35), "1":0.5*(0.35+0.5), "2":0.5*(0.5+0.65)}
    
    # Multiplicative bias due to shear and photometric redshifts
    Am_dic = {"0":1.021, "1":1.14, "2":1.16}

    Sigma_crit_dic = {"l3_z0":2.558834359999999819e-04, "l4_z0":2.562248809999999950e-04, 
                      "l5_z0":2.559633569999999851e-04, "l6_z0":2.557816409999999962e-04, 
                      "l3_z1":2.631864830000000188e-04, "l4_z1":2.640424589999999859e-04, 
                      "l5_z1":2.630129280000000004e-04, "l6_z1":2.629873459999999808e-04, 
                      "l3_z2":2.376251149999999903e-04, "l4_z2":2.379116389999999873e-04, 
                      "l5_z2":2.372028870000000031e-04, "l6_z2":2.373956570000000036e-04}
    return mass_dic[lam_z_bin], c_dic[lam_z_bin], B_dic[lam_z_bin], Rs_dic[lam_z_bin], z_dic[str(z_bin)], Am_dic[str(z_bin)], Sigma_crit_dic[lam_z_bin], profile_dic[lam_z_bin], Sigma_dic[lam_z_bin], sigRs_dic[lam_z_bin], sigB_dic[lam_z_bin], lam_z_bin

############################################################
############################################################
if __name__ == "__main__":
    output_loc = '/global/u2/t/titus/Titus/Lensing/output/'
    nbody_loc = '/global/u2/t/titus/Titus/Lensing/data/'
    lensing_loc = '/global/u2/t/titus/Titus/Lensing/data/McClintock_data/desy1_tamas/'
    filepath = '/pscratch/sd/t/titus/MiniUchuu/mcmc_results/'
    # sac_loc = '/global/u2/t/titus/Titus/Lensing/data/McClintock_data/desy1_SAC/'
    # plot_loc = '/global/u2/t/titus/Lensing/plots/'
    
    saved_ds_profiles_dic = nbody_loc+"saved_mini_uchuu_profiles_newdata_dic.npy"
    converted_mini_uchuu_dic = {} # a dictionary for mini Uchuu profiles after adding systematics to them
    saved_boost_profiles_dic = nbody_loc+"saved_boost_profiles_newdata_dic.npy"
    boost_dic = {} # boost factor profile computed from M19; to be used as input data in the MCMC
    ########################################################
    
    # Set cosmology (colossus)
    cosmo_params = {'flat': True, 'H0': 67.74, 'Om0': 0.3089, 'Ob0': 0.0486, 'sigma8': 0.8159, 'ns': 0.9667}
    cosmology.addCosmology('MiniUchuu', cosmo_params)
    cosmo = cosmology.setCosmology('MiniUchuu')
    h = cosmo_params['H0']/100 
    
    z=0.3
    rmu = ReadMiniUchuu(nbody_loc, redshift=z)
    xp, yp, zp = rmu.read_particles()

    #### halos ####
    rmu.read_halos(Mmin=1e13)
    boxsize = rmu.boxsize
    xh = rmu.xh
    yh = rmu.yh
    zh = rmu.zh
    mass = rmu.mass
    rhocrit = 2.775e11
    OmegaM = rmu.OmegaM
    mpart = rhocrit*OmegaM*boxsize**3/len(xp)

    # out_loc = 'data/'
    # ml = MeasureLensing(nbody_loc, Rmin=0.0323, Rmax=30, pimax=100)
    ml = MeasureLensing(nbody_loc, Rmin=-2, Rmax=2.3, pimax=100)
    ml.write_bin_file()
    
    # compute richness
    true_mass = mass#[sel]

    
    ##############################################
    # Melchior, 2017 scaling relation
    lam_0 = 30 # Pivot richness
    z = 0.30 # redshift
    z_0 = 0.5 # Pivot redshift
    G_z = 0.18 # Redshift scaling index
    F_lam = 1.12 # Richness scaling index
    q_ln_lam = 0.25 #scatter
    Mpivot = 10**14.371 #pivot mass
    converted_M = mass*h#[true_mass>=1e14] # mass

    fiducial_C = 0.0 # just a random guess
    fiducial_alpha = 1/F_lam
    fiducial_pi = -(1/F_lam)*(np.log(Mpivot) + G_z*np.log((1+z)/(1+z_0)) - F_lam*np.log(lam_0))
    fiducial_richness = fiducial_alpha*np.log(converted_M) + fiducial_pi #ln_lam

    # fiducial_richness = ln_lam #norm.rvs(ln_lam, 0.25) #np.exp(ln_lam)
    C_list = [0.0]
    M_input = np.log(mass*h)

    seed = 6
    np.random.seed(seed)
    random_states = np.random.randint(0, 100, size=3)

    for i,random_state in enumerate(random_states):
    # for i in range(2):
        # new_richness, alpha, pi, C, ln_mass, ln_lam, rsquared, reg_line, x_cod_new, y_cod_new, z_cod_new = evrard_extra_term(fiducial_richness, M_input, fiducial_C, xh, yh, zh)
        new_richness, alpha, pi, C, rsquared, ln_mass, all_mass = evrard_extra_term(fiducial_richness, M_input, fiducial_C, random_state)
        fiducial_C = C
        fiducial_alpha = alpha
        fiducial_pi = pi
        fiducial_richness = new_richness
        
        new_richness, alpha, pi, C, rsquared, ln_mass = evrard_extra_term2(fiducial_richness, all_mass, fiducial_C)
        fiducial_C = C
        fiducial_alpha = alpha
        fiducial_pi = pi
        fiducial_richness = new_richness
        C_list.append(C)
        
    ##############################################
    ##############################################
    richness = np.exp(new_richness)

    # Creating richness bins
    sel0 = (richness >= 20)&(richness < 30)
    sel1 = (richness >= 30)&(richness < 45)
    sel2 = (richness >= 45)&(richness < 60)
    sel3 = richness >= 60

    lam_bin0 = richness[sel0]
    lam_bin1 = richness[sel1]
    lam_bin2 = richness[sel2]
    lam_bin3 = richness[sel3]
    
    mean_lam0 = np.mean(lam_bin0)
    mean_lam1 = np.mean(lam_bin1)
    mean_lam2 = np.mean(lam_bin2)
    mean_lam3 = np.mean(lam_bin3)

    print(f'Length of richness bin 1: {len(lam_bin0)}')
    print(f'Length of richness bin 2: {len(lam_bin1)}')
    print(f'Length of richness bin 3: {len(lam_bin2)}')
    print(f'Length of richness bin 4: {len(lam_bin3)}\n')

    # Mean mass of each bin
    # mass_true = mass[sel]
    mean_mass_true0 = np.mean(mass[sel0])
    mean_mass_true1 = np.mean(mass[sel1])
    mean_mass_true2 = np.mean(mass[sel2])
    mean_mass_true3 = np.mean(mass[sel3])

    print('mean mass 1:', f'{mean_mass_true0:e}')
    print('mean mass 2:', f'{mean_mass_true1:e}')
    print('mean mass 3:', f'{mean_mass_true2:e}')
    print('mean mass 4:', f'{mean_mass_true3:e}\n')

    rp_, Sigma0, DeltaSigma0 = ml.measure_lensing(xh[sel0], yh[sel0], zh[sel0], xp, yp, zp, boxsize, mpart)
    rp_, Sigma1, DeltaSigma1 = ml.measure_lensing(xh[sel1], yh[sel1], zh[sel1], xp, yp, zp, boxsize, mpart)
    rp_, Sigma2, DeltaSigma2 = ml.measure_lensing(xh[sel2], yh[sel2], zh[sel2], xp, yp, zp, boxsize, mpart)
    rp_, Sigma3, DeltaSigma3 = ml.measure_lensing(xh[sel3], yh[sel3], zh[sel3], xp, yp, zp, boxsize, mpart)
    print('DeltaSigma1:', DeltaSigma0[:15], '\n')
    print('DeltaSigma2:', DeltaSigma1[:15], '\n')
    print('DeltaSigma3:', DeltaSigma2[:15], '\n')
    print('DeltaSigma4:', DeltaSigma3[:15], '\n')
    
    #Start by specifying the cosmology
    Omega_b = cosmo_params['Ob0']
    Omega_m = cosmo_params['Om0']
    Omega_cdm = Omega_m - Omega_b
    sigma8 = cosmo_params['sigma8']
    A_s = 2.1e-9 #np.exp(3.064)/1e10 NB: ln(1e10*As)=3.064
    n_s = cosmo_params['ns']
    # z = 0.3
    
    #Create a params dictionary
    #Need to specify the max wavenumber
    k_max = 1000 #UNITS: 1/Mpc

    params = {
             'output':'mPk',
             'h':h,
             'sigma8':sigma8,
             'n_s':n_s,
             'Omega_b':Omega_b,
             'Omega_cdm':Omega_cdm,
             'P_k_max_1/Mpc':k_max,
             'z_max_pk':1.0, #Default value is 10
             'non linear':'halofit',
    }

    #Initialize the cosmology and compute everything
    cosmo_ctk = Class()
    cosmo_ctk.set(params)
    cosmo_ctk.compute()
    
    ##########################################################
    parser = argparse.ArgumentParser(description='Command-line arguments.')
    parser.add_argument('--redshift', type=int, default=0, help='Redshift bin')
    parser.add_argument('--start', type=int, default=0, help='Starting point of the loop')
    parser.add_argument('--end', type=int, default=4, help='End point of the loop')
    parser.add_argument('--sys_name', type=str, default="_sys0", help='Give file name')
    # parser.add_argument('--precision', type=float, default=0.0005, help='PC termination precision criterion.')
    # parser.add_argument('--model_name', type=str, default='FLCDM', help='Model names')
    # parser.add_argument('--data_name', type=str, default='UHzB', help='Data names')
    # parser.add_argument('--default', action='store_true', help='Default PC sample settings')
    args = parser.parse_args()
    i = args.redshift
    start = args.start
    end = args.end
    sys_name = args.sys_name
    
    ##########################################################
    for j in range(start,end):
        print(f'Richness bin {j+1} in progress...')
        if j+3 == 3:
            low = 20
            high = 30
            mass_orig, c_orig, B0_orig, Rs_orig, z, Am_orig, Sigma_crit_inv0, profile_ds, Sigma_data0, sigRs, sigB0, lam_z_bin = getargs(j+3,i)
            # mass, concentration, B0, Rs, tau (miscentering offset), fmis, Am.
            true_params = np.array([np.log10(mass_orig), c_orig, B0_orig, Rs_orig, 0.17, 0.25, Am_orig])  
            readerfile = filepath+f"newdata_sigboosts{sys_name}_Fig9_mcmc_results_l"+str(j+3)+"_z"+str(i)+".h5"
        elif j+3 == 4:
            low = 30
            high = 45
            mass_orig, c_orig, B0_orig, Rs_orig, z, Am_orig, Sigma_crit_inv0, profile_ds, Sigma_data0, sigRs, sigB0, lam_z_bin = getargs(j+3,i)
            # mass, concentration, B0, Rs, tau (miscentering offset), fmis, Am.
            true_params = np.array([np.log10(mass_orig), c_orig, B0_orig, Rs_orig, 0.17, 0.25, Am_orig])  
            readerfile = filepath+f"newdata_sigboosts{sys_name}_Fig9_mcmc_results_l"+str(j+3)+"_z"+str(i)+".h5"
        elif j+3 == 5:
            low = 45
            high = 60
            mass_orig, c_orig, B0_orig, Rs_orig, z, Am_orig, Sigma_crit_inv0, profile_ds, Sigma_data0, sigRs, sigB0, lam_z_bin = getargs(j+3,i)
            # mass, concentration, B0, Rs, tau (miscentering offset), fmis, Am.
            true_params = np.array([np.log10(mass_orig), c_orig, B0_orig, Rs_orig, 0.17, 0.25, Am_orig]) 
            readerfile = filepath+f"newdata_sigboosts{sys_name}_Fig9_mcmc_results_l"+str(j+3)+"_z"+str(i)+".h5"
        elif j+3 == 6:
            low = 60
            high = 2000 # Infinity
            mass_orig, c_orig, B0_orig, Rs_orig, z, Am_orig, Sigma_crit_inv0, profile_ds, Sigma_data0, sigRs, sigB0, lam_z_bin = getargs(j+3,i)
            # mass, concentration, B0, Rs, tau (miscentering offset), fmis, Am.
            true_params = np.array([np.log10(mass_orig), c_orig, B0_orig, Rs_orig, 0.17, 0.25, Am_orig])  
            readerfile = filepath+f"newdata_sigboosts{sys_name}_Fig9_mcmc_results_l"+str(j+3)+"_z"+str(i)+".h5"

        ########################################
        if i == 0:
            tau_mu_prior, fmis_mu_prior, Am_mu_prior = 0.17, 0.25, 1.021
            tau_sigma_prior, fmis_sigma_prior, Am_sigma_prior = 0.04, 0.08, 0.025
        elif i == 1:
            tau_mu_prior, fmis_mu_prior, Am_mu_prior = 0.17, 0.25, 1.014
            tau_sigma_prior, fmis_sigma_prior, Am_sigma_prior = 0.04, 0.08, 0.024
        elif i == 2:
            tau_mu_prior, fmis_mu_prior, Am_mu_prior = 0.17, 0.25, 1.016
            tau_sigma_prior, fmis_sigma_prior, Am_sigma_prior = 0.04, 0.08, 0.025
        ########################################
        
        # Data from McClintock et. al., 2019.
        file_DS = lensing_loc+"full-unblind-v2-mcal-zmix_y1subtr_l"+str(j+3)+"_z"+str(i)+"_profile.dat"
        profile = np.loadtxt(file_DS)

        dst_cov = np.loadtxt(lensing_loc+"full-unblind-v2-mcal-zmix_y1subtr_l"+str(j+3)+"_z"+str(i)+"_dst_cov.dat")
        # dst_cov = np.loadtxt(sac_loc+"SAC_z"+str(i)+"_l"+str(j+3)+".txt")      

        # Richness
        redmapper = fitsio.read(nbody_loc+"redmapper_y1a1_public_v6.4_catalog.fits")
        richness = redmapper["LAMBDA"]
        bin_sel = (richness >= low) & (richness < high)
        lam_bin = richness[bin_sel]
        mean_lam = np.mean(lam_bin)
        print("Mean richness", mean_lam)
        # print(len(lam_bin))
        z_lambda = redmapper["Z_LAMBDA"]
        z_lambda_bin = z_lambda[bin_sel]
        z_mean = np.mean(z_lambda_bin)
        
        #DeltaSigma profile with systematics incorporated
        # rp = rp_
        # boost_cov = np.zeros((len(boost_cov0), len(boost_cov0)))
        # np.fill_diagonal(boost_cov, np.diag(boost_cov0))
        
        # dst_cov = np.zeros((len(dst_cov0), len(dst_cov0)))
        # np.fill_diagonal(dst_cov, np.diag(dst_cov0))
        # dst_cov = np.identity(len(dst_cov0))
        
        ##################################################################
        # Radial bins
        rp = np.array([ 0.04221139,  0.06651455,  0.10516441,  0.16537565,  0.26069825,
                       0.41202975,  0.65045543,  1.02563815,  1.61857299,  2.55379112, 
                       4.02731114,  6.34991963, 10.00980414, 15.784786  , 24.87216972])
        ds, boost_data = add_sys_to_data(mass_orig, profile_ds, Sigma_data0, Rs_orig, B0_orig, Am_orig, z, mean_lam, rp)
        # converted_mini_uchuu_dic[lam_z_bin] = ds
        # boost_dic[lam_z_bin] = boost_data

        # print("ds", ds)
        # print("shape", ds.shape)
        
    # np.save(saved_ds_profiles_dic, converted_mini_uchuu_dic)
    # np.save(saved_boost_profiles_dic, boost_dic)
        
        #------------------Boost factor covariance computed via error propagation of the boost factor equation
        boost_dx = lambda x: ctk.boostfactors.boost_nfw_at_R(rp, x, Rs_orig)
        boost_dy = lambda y: ctk.boostfactors.boost_nfw_at_R(rp, B0_orig, y)

        f_dx = nd.Derivative(boost_dx)
        f_dy = nd.Derivative(boost_dy)
        sigboost = np.sqrt((f_dx(B0_orig)*sigB0)**2 + (f_dy(Rs_orig)*sigRs)**2)
        boost_cov = np.diag(np.full(len(rp), 1/sigboost**2))
        
        #################################################################
        # Initial guess for the MCMC
        start_params = np.array([14.20, 4.5, 0.3, 0.7, 0.15, 0.23, 1.0]) # mass, concentration, B0, Rs, tau (miscentering offset), fmis, Am.
        nsteps = 10000
        burnin = int(0.1*nsteps)
        sampler, flat_samples, samples_n, O_mcmc = run_mcmc(data = ds, params = start_params, 
                                                            nwalkers = 32, nsteps = nsteps, burnin = burnin, 
                                                            sac = dst_cov, boost_data = boost_data, z=z, 
                                                            readerfile = readerfile, lam = mean_lam, 
                                                            boost_cov = boost_cov)
    
    
    # Added the factor of h*(1+z) to all Rs in boost data, boost cov, Redges, except boost model, radial cut at 0.1Mpc.
    # mpirun -np 8 python mcclintockFig9_newdata_sigboost_new_changed_bcov_phys_units.py --redshift 0 --start 0 --end 4 --sys_name _fullrun_mh_no_h1z_in_bmodel_only_rs_old_prior_setup_changed_bov_phys_units
    