"""
Duplication of the vector fitting algorithm in python (http://www.sintef.no/Projectweb/VECTFIT/)

All credit goes to Bjorn Gustavsen for his MATLAB implementation, and the following papers


 [1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency
     domain responses by Vector Fitting", IEEE Trans. Power Delivery,
     vol. 14, no. 3, pp. 1052-1061, July 1999.

 [2] B. Gustavsen, "Improving the pole relocating properties of vector
     fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
     July 2006.

 [3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
     "Macromodeling of Multiport Systems Using a Fast Implementation of
     the Vector Fitting Method", IEEE Microwave and Wireless Components
     Letters, vol. 18, no. 6, pp. 383-385, June 2008.

updated for Python 3 - RXA254, July-2021

"""
__author__ = 'Phil Reinhold'

import numpy as np
import matplotlib.pyplot as plt

#from pylab import *
from numpy.linalg import eigvals, lstsq


plt.style.use('BodePlot.mplstyle')

def cc(z):
    return z.conjugate()

def model(s, poles, residues, d, h):
    return sum(r/(s-p) for p, r in zip(poles, residues)) + d + s*h

def vectfit_step(f, s, poles):
    """
    f = complex data to fit
    s = j*frequency
    poles = initial poles guess
        note: All complex poles must come in sequential complex conjugate pairs
    returns adjusted poles
    """
    N  = len(poles)
    Ns = len(s)

    cindex = np.zeros(N)
    # cindex is:
    #   - 0 for real poles
    #   - 1 for the first of a complex-conjugate pair
    #   - 2 for the second of a cc pair
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                assert cc(poles[i]) == poles[i+1], ("Complex poles must come in conjugate pairs: %s, %s" % (poles[i], poles[i+1]))
                cindex[i] = 1
            else:
                cindex[i] = 2

    # First linear equation to solve. See Appendix A (of what?)
    A = np.zeros((Ns, 2*N+2), dtype=np.complex64)
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1/(s - p)
        elif cindex[i] == 1:
            A[:, i] = 1/(s - p) + 1/(s - cc(p))
        elif cindex[i] == 2:
            A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

        A [:, N+2+i] = -A[:, i] * f

    A[:, N]   = 1
    A[:, N+1] = s

    # Solve Ax == b using pseudo-inverse
    b = f
    A = np.vstack((np.real(A), np.imag(A)))
    b = np.concatenate((np.real(b), np.imag(b)))
    x, residuals, rnk, s = lstsq(A, b, rcond=-1)

    residues = x[:N]
    d = x[N]
    h = x[N+1]

    # We only want the "tilde" part in (A.4)
    x = x[-N:]

    # Calculation of zeros: Appendix B
    A = np.diag(poles)
    b = np.ones(N)
    c = x
    for i, (ci, p) in enumerate(zip(cindex, poles)):
        if ci == 1:
            x, y = np.real(p), np.imag(p)
            A[i, i]     =  x
            A[i+1, i+1] =  x
            A[i, i+1]   = -y
            A[i+1, i]   =  y
            b[i]        =  2
            b[i+1]      =  0
            #cv = c[i]
            #c[i,i+1] = np.real(cv), np.imag(cv)

    H = A - np.outer(b, c)
    H = np.real(H)
    new_poles = np.sort(eigvals(H))
    unstable  = np.real(new_poles) > 0
    new_poles[unstable] -= 2*np.real(new_poles)[unstable]
    return new_poles

# Dear gods of coding style, I sincerely apologize for the following copy/paste
def calculate_residues(f, s, poles, rcond=-1):
    Ns = len(s)
    N  = len(poles)

    cindex = np.zeros(N)
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                assert cc(poles[i]) == poles[i+1], ("Complex poles must come in conjugate pairs: %s, %s" % poles[i:i+1])
                cindex[i] = 1
            else:
                cindex[i] = 2

    # use the new poles to extract the residues
    A = np.zeros((Ns, N+2), dtype=np.complex128)
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1/(s - p)
        elif cindex[i] == 1:
            A[:, i] = 1/(s - p) + 1/(s - cc(p))
        elif cindex[i] == 2:
            A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

    A[:, N]   = 1
    A[:, N+1] = s
    # Solve Ax == b using pseudo-inverse
    b  = f
    A  = np.vstack((np.real(A), np.imag(A)))
    b  = np.concatenate((np.real(b), np.imag(b)))
    cA = np.linalg.cond(A)
    if cA > 1e13:
        print('Warning!: Ill Conditioned Matrix. Consider scaling the problem down')
        print('Cond(A)', cA)
    x, residuals, rnk, s = lstsq(A, b, rcond=rcond)

    # Recover complex values
    x = np.complex64(x)
    for i, ci in enumerate(cindex):
       if ci == 1:
           r1, r2 = x[i:i+2]
           x[i]   = r1 - 1j*r2
           x[i+1] = r1 + 1j*r2

    residues = x[:N]
    d = x[N].real
    h = x[N+1].real
    return residues, d, h

def print_params(poles, residues, d, h):
    cfmt = "{0.real:g} + {0.imag:g}j"
    print("poles: " + ", ".join(cfmt.format(p) for p in poles))
    print("residues: " + ", ".join(cfmt.format(r) for r in residues))
    print("offset: {:g}".format(d))
    print("slope: {:g}".format(h))

def vectfit_auto(f, s, n_poles = 10, n_iter = 10, printparams = False,
                 inc_real = False, loss_ratio = 1e-2, rcond = -1, track_poles = False):

    w          = np.imag(s)
    pole_locs  = np.linspace(w[0], w[-1], n_poles+2)[1:-1]
    lr         = loss_ratio
    init_poles = poles = np.concatenate([[p*(-lr + 1j), p*(-lr - 1j)] for p in pole_locs])

    if inc_real:
        poles = np.concatenate((poles, [1]))

    poles_list = []
    for _ in range(n_iter):
        poles = vectfit_step(f, s, poles)
        poles_list.append(poles)

    residues, d, h = calculate_residues(f, s, poles, rcond=rcond)

    if track_poles:
        return poles, residues, d, h, np.array(poles_list)

    if printparams:
        print_params(poles, residues, d, h)

    return poles, residues, d, h


def vectfit_auto_rescale(f, s, printparams=False, **kwargs):
    s_scale = np.abs(s[-1])
    f_scale = np.abs(f[-1])

    if printparams:
        print('SCALED')
        
    poles_s, residues_s, d_s, h_s = vectfit_auto(f / f_scale, s / s_scale, **kwargs)
    poles = poles_s * s_scale
    residues = residues_s * f_scale * s_scale
    d = d_s * f_scale
    h = h_s * f_scale / s_scale

    if printparams:
        print('UNSCALED')
        print_params(poles, residues, d, h)
    return poles, residues, d, h




if __name__ == '__main__':
    test_s = 1j*np.linspace(1, 1e5, 800)
    test_poles = [
        -4500,
        -41000,
        -100+5000j, -100-5000j,
        -120+15000j, -120-15000j,
        -3000+35000j, -3000-35000j,
        -200+45000j, -200-45000j,
        -1500+45000j, -1500-45000j,
        -500+70000j, -500-70000j,
        -1000+73000j, -1000-73000j,
        -2000+90000j, -2000-90000j,
    ]
    test_residues = [
        -3000,
        -83000,
        -5+7000j, -5-7000j,
        -20+18000j, -20-18000j,
        6000+45000j, 6000-45000j,
        40+60000j, 40-60000j,
        90+10000j, 90-10000j,
        50000+80000j, 50000-80000j,
        1000+45000j, 1000-45000j,
        -5000+92000j, -5000-92000j
    ]
    test_d = .2
    test_h = 2e-5

    test_f  = sum(c/(test_s - a) for c, a in zip(test_residues, test_poles))
    test_f += test_d + test_h*test_s
    vectfit_auto(test_f, test_s)

    poles, residues, d, h = vectfit_auto_rescale(test_f, test_s)
    fitted = model(test_s, poles, residues, d, h)

    ff = test_s.imag / 2 / np.pi
    
    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].loglog(ff, np.abs(test_f), ls='', marker='.', label='Data')
    ax[0].loglog(ff, np.abs(fitted), alpha=0.3, label='Fit')
    ax[0].set_ylabel('Mag')
    ax[0].legend()
    
    ax[1].semilogx(ff, np.angle(test_f, deg=True),  ls='', marker='.')
    ax[1].semilogx(ff, np.angle(fitted, deg=True), alpha = 0.3)
    ax[1].set_ylabel('Phase [deg]')
    ax[1].set_xlabel('Frequency [Hz]')
    
    plt.savefig("test_vectfit.pdf", bbox_inches='tight')
    #plt.show()
