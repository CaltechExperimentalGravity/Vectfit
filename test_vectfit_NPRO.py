#!/usr/bin/env python
import pwd, os
import vectfit as vf
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from prettytable import PrettyTable

# convert from (f0, Q) to (a+1j*b, a-1j*b)
def get_res_g_pole_pair(f0, Q):
    """
    rr, ii are the real and imag parts of the complex pole
    """
    w0 = 2*np.pi * f0
    rr = -w0/2/Q
    ii = np.sqrt(w0**2 - rr**2)
    return rr, ii

def get_pp_from_resg(f0, Q):
    """
    complex conjugate pole pair for a resonance with f0 & Q
    """
    w0 = 2*np.pi * f0
    rr = -w0/2/Q
    ii = np.sqrt(w0**2 - rr**2)
    mypp = [rr1+1j*ii1, rr1-1j*ii1]
    return mypp

plt.style.use('BodePlot.mplstyle')

# Create some test data using known poles and residues
# Substitute your source of data as needed

# Note our independent variable lies along the imaginary axis
ff = np.logspace(3, 6, 400)
ww = 2 * np.pi * ff
s  = 1j * ww


# generate a model to simulate

# a collection of poles and zeros
rr1, ii1 = get_res_g_pole_pair(13e3, 7)
rr2, ii2 = get_res_g_pole_pair(37e3, 20)
rr3, ii3 = get_res_g_pole_pair(55e3, 33)
rr4, ii4 = get_res_g_pole_pair(57e3, 23)
rr5, ii5 = get_res_g_pole_pair(49e3, 15)

p = np.array([rr1+1j*ii1, rr1-1j*ii1, rr3+1j*ii3, rr3-1j*ii3, rr5+1j*ii5, rr5-1j*ii5])
z = np.array([rr2+1j*ii2, rr2-1j*ii2, rr4+1j*ii4, rr4-1j*ii4])
k = (rr1**2.+ii1**2.)*(rr3**2.+ii3**2.)*(rr5**2.+ii5**2.)/(rr2**2.+ii2**2.)/(rr4**2.+ii4**2.)



# this is a high pass filter
zz, pp, kk = sig.ellip(4, 6, 30, 2*np.pi*64e3, 'high', analog=True, output='zpk')
#rr4, ii4 = get_res_g_pole_pair(600, 23)
#rr5, ii5 = get_res_g_pole_pair(800, 10)

# multiply the TFs together
z = np.concatenate([z, zz])
p = np.concatenate([p, pp])
k *= kk

# this is the Transfer Function with no noise
ww, mytfdata = sig.freqs_zpk(z, p, k, worN=ww)
model = mytfdata.copy()



# add some noise to the measurement
# Gaussian in amplitude, uniform in phase
mmm = 1e-4
pho = np.random.uniform(-180, 180, len(ww))
maa = np.random.normal(loc = mmm, scale = mmm/5, size=len(ww))
nzz = maa * np.exp(1j * pho * np.pi/180)

upsilon = 1e-9 # this is to prevent divide by zero stuff
weight = np.abs(mytfdata) / (upsilon + maa)
weight /= np.max(weight)
# weight = weight**0.5

# I don't think this is used for anything yet, but it should be
coh = np.abs(mytfdata)**2 / (np.abs(nzz)**2 + np.abs(mytfdata)**2)

# inverting the above coherence eq to get SNR
SNR_data = coh / (1 - coh)

# maybe make the weighting more serious?
#SNR_data = SNR_data**2

# replace this with measured data
mytfdata += nzz # add noise to the noiseless TF



# d == offset, h == slope
#d = .2
#h = 2e-5
#vmod   = vectfit.model(s, test_poles, test_residues, d, h)

# Run algorithm, results hopefully match the known model parameters
t_0 = timer()

n_poles = 12 # number of initial poles

# no weight, no cleaning
poles, residues, d, h = vf.vectfit_auto_rescale(mytfdata, s, printparams=False, n_poles=n_poles)
# convert fraction expansion to zpk
zz, pp, kk = vf.to_zpk(poles, residues, d, h)

# (W) weight. Weight it by the amplitude SNR
p_w, r_w, d_w, h_w = vf.vectfit_auto_rescale(mytfdata, s, ww=SNR_data, printparams=False, n_poles=n_poles)
# convert fraction expansion to zpk
z_w, p_w, k_w = vf.to_zpk(p_w, r_w, d_w, h_w)
#print(len(p_w), len(p), len(z_w), len(z), k, k_w)


p_lowest = np.min(np.abs(p_w))


# discard zp pairs that are too similar
zpk_w = (z_w, p_w, k_w)
zpk_c = vf.discard_similar_pz(zpk_w, cut=p_lowest/100, f_match=p_lowest/100)
z_co, p_co, k_co = zpk_c
#print(" ")
#print("Cutoff ~co-located pole/zero pairs.")
#print(len(p_co), len(p), len(z_co), len(z), k, k_co)


# discard z's and p's that are in the low coherence regime
coh_cut = 0.5
#zpk_c = (z_co, p_co, k_co)
zpk_c = vf.discard_features_low_coh(zpk_c, ff, coh, 
                             coh_cut = coh_cut, f_match = 1.)
z_c, p_c, k_c = zpk_c
#print(" ")
#print("Cutoff features in regions of coherence " + r"< {coh:0.1f}".format(coh = coh_cut) + ".")
#print(len(p_c), len(p), len(z_c), len(z), k, k_c)



t_elapsed = timer() - t_0
print('Elapsed time = {t:0.2f} seconds.'.format(t = t_elapsed))

tab = PrettyTable(['Fit Type', '# of poles', "# of zeros"])
tab.add_row(['Model',                  len(p),    len(z)])
tab.add_row(['no weight',              len(pp),   len(zz)])
tab.add_row(['SNR weight',             len(p_w),  len(z_w)])
tab.add_row(['SNR weight + colocated', len(p_co), len(z_co)])
tab.add_row(['SNR + co + cleaning',    len(p_c),  len(z_c)])
print(tab)


__, fitted   = sig.freqs_zpk(zz,  pp,  kk, worN=ww)
__, fitted_w = sig.freqs_zpk(z_w, p_w, k_w, worN=ww)
__, fitted_c = sig.freqs_zpk(z_c, p_c, k_c, worN=ww)


bb, aa = sig.invres(r_w, p_w, d_w)
zz, pp, kk = sig.tf2zpk(bb,aa)
__, fitted_w_zpk = sig.freqs_zpk(zz, pp, kk, worN=ww)


# Calculate Errors ~~~~~~~~~~~~~~~~~~~~~~
err   = mytfdata/fitted - 1
err_w = mytfdata/fitted_w - 1

# fractional error of fit w.r.t. model
err_c = fitted_c/model - 1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ===============================================================
# plot the results of the simulation and the fits and residuals
fig,ax = plt.subplots(3,1,sharex=True, figsize=(10,10),
                      gridspec_kw={'height_ratios': [1, 0.5, 0.3]})


ax[0].loglog(ff, np.abs(mytfdata),
        ls='', marker='.', alpha=0.3, color='xkcd:Burple', label='Data')
ax[0].loglog(ff, np.abs(fitted),
        alpha=0.5,
    label='Fit (no weight; %i poles, %i zeros)'%(len(pp), len(zz)))
ax[0].loglog(ff, np.abs(fitted_w),
        alpha=0.5,
        label='Fit (w weight; %i poles, %i zeros)'%(len(p_w), len(z_w)))
ax[0].loglog(ff, np.abs(fitted_c),
        alpha=0.9, label='Fit (w weight+discarding; %i poles, %i zeros)'%(len(p_c), len(z_c)))
ax[0].loglog(ff, np.abs(model),
        alpha=0.7, ls='--', color='xkcd:Black',
             label='Model (%i poles, %i zeros)'%(len(p), len(z)))
# ax[0].loglog(ff, np.abs(err), alpha=0.3, label='Residual (no)')
ax[0].loglog(ff, np.abs(err_c),
        ls='', marker='.', alpha=0.3,
             label='Residual (Fit/Model - 1)')
ax[0].set_ylabel('Mag')
ax[0].legend()

ax[1].semilogx(ff, np.angle(mytfdata, deg=True),
               ls='', marker='.', alpha=0.3, color='xkcd:Burple')
ax[1].semilogx(ff, np.angle(fitted,   deg = True), alpha = 0.5)
ax[1].semilogx(ff, np.angle(fitted_w, deg = True), alpha = 0.5)
ax[1].semilogx(ff, np.angle(fitted_c, deg = True), alpha = 0.9)
ax[1].semilogx(ff, np.angle(model, deg=True),
               alpha=0.7, ls='--',color='xkcd:Black')
# ax[1].semilogx(ff, np.angle(err,   deg=True), alpha = 0.3)
ax[1].semilogx(ff, np.angle(err_c, deg=True),
               ls='', marker='.', alpha = 0.3)
ax[1].set_yticks([-180, -90, 0, 90, 180])
ax[1].set_ylabel('Phase [deg]')


ax[2].semilogx(ff, coh, alpha=0.7, color='xkcd:Burple')
ax[2].set_ylabel('Coherence')
ax[2].set_xlabel('Frequency [Hz]')



metadata = {}
metadata['Title']    = 'Vectfit Example'
metadata['Author']   = pwd.getpwuid(os.getuid())[0]
#metadata['Subject']  = 'How to add metadata to a PDF file within matplotlib'
#metadata['Keywords'] = 'PdfPages example'
#metadata['InitialView'] = 'Full Page'

#plt.show()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("test_vectfit.pdf", bbox_inches='tight',
            metadata = metadata)
