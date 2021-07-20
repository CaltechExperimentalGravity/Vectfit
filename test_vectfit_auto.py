import vectfit as vf
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from timeit import default_timer as timer


plt.style.use('BodePlot.mplstyle')

# Create some test data using known poles and residues
# Substitute your source of data as needed

# Note our independent variable lies along the imaginary axis
ff = np.logspace(1, 4, 1000)
ww = 2 * np.pi * ff
s  = 1j * ww

z, p, k = sig.ellip(4, 5, 40, Wn=100*(2*np.pi), btype='low', analog=True, output='zpk')

ww, mytfdata = sig.freqs_zpk(z, p, k, worN=ww)

# add some noise to the measurement
mmm = 1e-3
pho = np.random.uniform(-180, 180, len(ww))
maa = np.random.normal(loc = mmm, scale = mmm/5, size=len(ww))
nzz = maa * np.exp(1j * pho * np.pi/180)

mytfdata += nzz

# d == offset, h == slope
#d = .2
#h = 2e-5
#vmod   = vectfit.model(s, test_poles, test_residues, d, h)

# Run algorithm, results hopefully match the known model parameters
t_0 = timer()

poles, residues, d, h = vf.vectfit_auto_rescale(mytfdata, s, printparams=False, n_poles=5)

t_elapsed = timer() - t_0
print('Elapsed time = {t:0.2f} seconds.'.format(t=t_elapsed))


fitted = vf.model(s, poles, residues, d, h)

fig,ax = plt.subplots(2,1,sharex=True)

err = mytfdata/fitted - 1

ax[0].loglog(ff, np.abs(mytfdata), ls='', marker='.', label='Data')
ax[0].loglog(ff, np.abs(fitted), alpha=0.3, label='Fit')
ax[0].loglog(ff, np.abs(err), alpha=0.3, label='Residual')
ax[0].set_ylabel('Mag')
ax[0].legend()

ax[1].semilogx(ff, np.angle(mytfdata, deg=True),  ls='', marker='.')
ax[1].semilogx(ff, np.angle(fitted,   deg=True), alpha = 0.3)
ax[1].semilogx(ff, np.angle(err,   deg=True), alpha = 0.3)
ax[1].set_ylabel('Phase [deg]')
ax[1].set_xlabel('Frequency [Hz]')

plt.savefig("test_vectfit.pdf", bbox_inches='tight')
