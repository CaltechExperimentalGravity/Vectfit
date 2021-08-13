#!/usr/bin/env python

import vectfit as vf
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# convert from (f0, Q) to (a+1j*b, a-1j*b)
def get_res_g_pole_pair(f0, Q):
    """
    rr, ii are the real and imag parts of the complex pole
    """
    w0=2.*np.pi*f0
    rr=-w0/2./Q
    ii=np.sqrt(w0**2.-rr**2.)
    return rr, ii


# plt.style.use('BodePlot.mplstyle')

# Create some test data using known poles and residues
# Substitute your source of data as needed

# Note our independent variable lies along the imaginary axis
ff = np.logspace(1, 4, 1000)
ww = 2 * np.pi * ff
s  = 1j * ww

# z, p, k = sig.ellip(4, 5, 40, Wn=100*(2*np.pi), btype='low', analog=True, output='zpk')
# z, p, k = sig.cheby1(4, 3, Wn=500*(2*np.pi), btype='low', analog=True, output='zpk')
rr1, ii1 = get_res_g_pole_pair(300, 25)
rr2, ii2 = get_res_g_pole_pair(300, 6)
rr3, ii3 = get_res_g_pole_pair(100, 10)
rr4, ii4 = get_res_g_pole_pair(600, 3)

p = np.array([rr1+1j*ii1, rr1-1j*ii1, rr3+1j*ii3, rr3-1j*ii3])
z = np.array([rr2+1j*ii2, rr2-1j*ii2, rr4+1j*ii4, rr4-1j*ii4])
k = (rr1**2.+ii1**2.)*(rr3**2.+ii3**2.)/(rr2**2.+ii2**2.)/(rr4**2.+ii4**2.)



ww, mytfdata = sig.freqs_zpk(z, p, k, worN=ww)
mytfdata_clean = mytfdata.copy()

# add some noise to the measurement
mmm = 1e-2
pho = np.random.uniform(-180, 180, len(ww))
maa = np.random.normal(loc = mmm, scale = mmm/5, size=len(ww))
nzz = maa * np.exp(1j * pho * np.pi/180)

weight = np.abs(mytfdata) / (1.e-6 + maa)
weight /= np.max(weight)
# weight = weight**0.5

mytfdata += nzz

# d == offset, h == slope
#d = .2
#h = 2e-5
#vmod   = vectfit.model(s, test_poles, test_residues, d, h)

# Run algorithm, results hopefully match the known model parameters
t_0 = timer()

# no weight
poles, residues, d, h = vf.vectfit_auto_rescale(mytfdata, s, printparams=False, n_poles=6)
# convert fraction expansion to zpk
zz, pp, kk = vf.to_zpk(poles, residues, d, h)

# w weight
p_w, r_w, d_w, h_w = vf.vectfit_auto_rescale(mytfdata, s, ww=weight, printparams=False, n_poles=6)
# convert fraction expansion to zpk
z_w, p_w, k_w = vf.to_zpk(p_w, r_w, d_w, h_w)


t_elapsed = timer() - t_0
print('Elapsed time = {t:0.2f} seconds.'.format(t=t_elapsed))


__, fitted = sig.freqs_zpk(zz, pp, kk, worN=ww)
__, fitted_w = sig.freqs_zpk(z_w, p_w, k_w, worN=ww)


bb, aa= sig.invres(r_w, p_w, d_w)
zz, pp, kk = sig.tf2zpk(bb,aa)
__, fitted_w_zpk = sig.freqs_zpk(zz, pp, kk, worN=ww)


fig,ax = plt.subplots(2,1,sharex=True)

err = mytfdata/fitted - 1
err_w = mytfdata/fitted_w - 1

ax[0].loglog(ff, np.abs(mytfdata), ls='', marker='.', label='Data')
ax[0].loglog(ff, np.abs(mytfdata_clean), alpha=0.7, ls='--', label='Model')
ax[0].loglog(ff, np.abs(fitted), alpha=0.6, label='Fit (no weight)')
ax[0].loglog(ff, np.abs(fitted_w), alpha=0.6, label='Fit (w weight)')
# ax[0].loglog(ff, np.abs(err), alpha=0.3, label='Residual (no)')
# ax[0].loglog(ff, np.abs(err_w), alpha=0.3, label='Residual (w)')
ax[0].set_ylabel('Mag')
ax[0].legend()

ax[1].semilogx(ff, np.angle(mytfdata, deg=True),  ls='', marker='.')
ax[1].semilogx(ff, np.angle(mytfdata_clean, deg=True), alpha=0.7, ls='--')
ax[1].semilogx(ff, np.angle(fitted,   deg=True), alpha = 0.6)
ax[1].semilogx(ff, np.angle(fitted_w,   deg=True), alpha = 0.6)
# ax[1].semilogx(ff, np.angle(err,   deg=True), alpha = 0.3)
# ax[1].semilogx(ff, np.angle(err_w, deg=True), alpha = 0.3)

ax[1].set_ylabel('Phase [deg]')
ax[1].set_xlabel('Frequency [Hz]')

plt.show()
# plt.savefig("test_vectfit.pdf", bbox_inches='tight')
