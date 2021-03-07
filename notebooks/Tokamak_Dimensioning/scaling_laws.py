"""
Tokamak scaling laws
"""
import numpy as np
from scipy.constants import pi, e, k, mu_0
from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt

# Constant terms
C_loss = 6 * pi**2 * 1e19 * 1e-3 * e # for W in MW.s
C_fus = 17.59 * e * 1.18e-24 * 1e38 * pi**2 / 2  # for P_fus in MW
#C_fus = 9.51*pi**2*e*1e14 # ?? Yanick
C_beta = 4e2 * mu_0 * 1e19 * 1e3 * e  # beta expressed in %
C_n = 10/pi  # for n in 10^19 m^-3
C_I = 2 * pi * 1e-6 / mu_0  # for I in MA
ratio = 4.94  # ratio Pfus/Palpha (=5 w/o relativistic effects; =4.94 w/ relativistic effects)

print(f'C_loss={C_loss}')  # OK
print(f'C_fus={C_fus}')  # OK
print(f'C_beta={C_beta}')  # OK
print(f'C_n={C_n}')  # OK
print(f'C_I={C_I}')  # OK

# Additional parameters
falpha = 0.1  # fraction of alpha particles
fp     = 1.5  # peaking factor of the temperature profile (<p^2> = fp <p>^2)
thetai = 1/1.25  # Ratio of Te & Ti temperatures

# Additional corrections from Johner FST 2001
def _C_I(epsilon=0.323, kappa=1.7, delta=0.33):
    return C_I*(1.17 - 0.65*epsilon)/(1 - epsilon**2)**2 * \
        0.5*(1 + kappa**2 * (1 + 2*delta**2 - 1.2*delta**3))

def _Meff(falpha=0.1):
    " Average ion mass [Atomic Mass Unit] "
    return (5 - 2*falpha)/(2*(1 - falpha))

def _Cfus(falpha=0.1, thetai=0.8, fp=1.5):
    return C_fus * (1 - 2*falpha)**2 * thetai**2 * fp

def _Closs(falpha=0.1, thetai=0.8):
    return C_loss * (1 + thetai - thetai*falpha)/2 

def _Cbeta(falpha=0.1, thetai=0.8):
    return C_beta * (1 + thetai - thetai*falpha)/2

Meff = _Meff()
C_loss = _Closs()
C_fus = _Cfus()
C_beta = _Cbeta()
C_I = _C_I()


# n.T.tau_e and beta_N expressions
def nTtau_fromQ(Q=10, lambd=ratio):
    " n.T.tau_e from eq (2.18) "
    return C_loss/C_fus * Q / (1 + Q/lambd)

# TODO : the following is not usefull later
def nTtau_fromRBBeta(M=2.7, kappa=1.7, epsilon=0.323, qa=3, n_N=0.85, R=6.2, B=5.3, beta_N=1.8):
    " n.T.tau_e from eq (2.20) "
    C_SL = 0.0562
    # (n*T*Tau)^0.31 
    _nTTau = C_SL * C_n**0.41 * C_I **0.96 * C_beta**0.38 * M**0.19 * kappa**0.09 * \
             epsilon**0.68 * qa**(-0.96) * n_N**0.41 * R**0.42 * B**0.73 * beta_N**(-0.38)
    return _nTTau**(1/0.31)

# beta_N expressions
def beta_N1(P_DT=410, kappa=1.7, epsilon=0.323, qa=3, R=6.2, B=5.3):
    " beta_N from expression (2.10) "
    beta_N = qa * C_beta * np.sqrt(P_DT /(R**3 * B**4))  / \
                     np.sqrt(C_fus * C_I**2 * kappa * epsilon**4)
    return beta_N

def beta_N2(Q=10, lambd=ratio, M=Meff, kappa=1.7, epsilon=0.323, qa=3, n_N=0.85, R=6.2, B=5.3):
    " beta_N from expression (2.20)"
    _nTtau = nTtau_fromQ(Q, lambd)
    numer = _nTtau**0.31
    C_SL = 0.0562
    denom = C_SL * C_n**0.41 * C_beta**0.96 * M**0.19 * kappa**0.09 * epsilon**0.68 * \
            qa**(-0.96) * n_N**0.41 * R**0.42 * B**0.73
    return (numer/denom)**(-1/0.38) * 1e5 # why x 1e5 to have the correct scale?




def plot_beta_N_curves(P_DT=410, Q=10, 
                       kappa=1.7, epsilon=0.323, qa=3, 
                       lambd=5, M=2.7, n_N=0.85):
    _B = np.linspace(3, 9, 101)
    _R = np.linspace(2, 10, 101)
    _RR, _BB = np.meshgrid(_R, _B)

    _beta_N1 = beta_N1(B=_BB, R=_RR, 
                     P_DT=P_DT, kappa=kappa, epsilon=epsilon, qa=qa)
    _beta_N2 = beta_N2(B=_BB, R=_RR, 
                     Q=Q, lambd=lambd, M=M, kappa=kappa, 
                      epsilon=epsilon, qa=qa, n_N=n_N)

    fig, ax = plt.subplots(figsize=(7,5))
    c1=ax.contour(_RR, _BB, _beta_N1, levels=[1.4, 1.6, 1.8,  2, 2.2],
                 alpha=0.5, cmap='copper')
    ax.clabel(c1, inline=1, fontsize=10, fmt='%.1f')
    c2=ax.contour(_RR, _BB, _beta_N2, levels=[1.4, 1.6,  1.8,  2, 2.2], 
                  linestyles='dashed', alpha=0.8)
    ax.clabel(c2, inline=1, fontsize=10, fmt='%.1f')
    ax.set_xlabel('R [m]')
    ax.set_ylabel('B [T]')
    ax.set_title(r'plain lines: $\beta_N(R,B)$ from (2.10)'+' \n'+ r' dashed lines: $\beta_N(R,B)$ from (2.20)')

    # determine the points where both beta_N are equals to the target (common points)
    # from the contour lines
    # beta_N_target = 1.8 -> 3rd line on contour
    RB_sol1 = c1.allsegs[2][0]
    RB_sol2 = c2.allsegs[2][0]
    B_sol1 = RB_sol1[:,1]
    B_sol2 = np.interp(RB_sol1[:,0], RB_sol2[:,0], RB_sol2[:,1])  # interpolate to be able to find the nearest value 

    R_sol = RB_sol1[np.argmin(np.abs(B_sol1 - B_sol2)), 0]
    B_sol = RB_sol1[np.argmin(np.abs(B_sol1 - B_sol2)), 1]

    ax.plot(R_sol, B_sol, '.', ms=20, color='k')

    ax.annotate(fr'(R,B) for $\beta_N=1.8$: ({R_sol:.2f} m, {B_sol:.2f} T)',  xy=(R_sol+.1, B_sol+.1), xytext=(5,8), arrowprops={'arrowstyle': '->'}, va='center')