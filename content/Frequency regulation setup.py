# Setup code for the "Frequency regulation nocode.ipynb" notebook.
# (this code is taken from the detailed notebook "Frequency regulation.ipynb")
# Pierre Haessig, september 2023, CC-BY http://creativecommons.org/licenses/by/4.0/

import numpy as np
from numpy import pi
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from ipywidgets import interact

# Constants
# Base frequency
f0 = 50 # Hz
Ω0 = 2*pi*f0

# Simulation duration: initial and final time
t_ini=-1 # s
t_fin=10 # s

### Simulation function
def sim_freq_response(ΔP_load, T_ΔP, H, s, T_fcr, fcr, fcr_lag):
    """
    Simulate frequency response from FCR

    Parameters:
    ΔP_load: excess of consumption (in per-unit)
    T_ΔP: duration of the excess of consumption (in s). can be np.inf
    H: inertia constant (s) >0
    s: FCR regulation inverse gain  (a few %) > 0 ("R" in Anderson 1990)
    T_fcr: FCR first order time constant (s) > 0

    Booleans:
    fcr: activation of FCR, True of False
    fcr_lag: activation of FCR response lag, True or False

    Returns t, f, P_fcr, f_lim
    """
    # Constants and parameters
    Sg = 1 # so that ΔP is in pu
    J = 2*H*Sg/Ω0**2
    
    if fcr:
        K = Sg/(f0*s)
        f_lim = f0 - ΔP_load/K
    else:
        K = 0.0
        f_lim = 0 # 0 Hz = blackout
    
    def fder(t, x):
        """ODE dynamics for grid with optional frequency regulation
        """
        f = x[0]
        Δf = f-f0
        Ω = 2*pi*f
        
        if fcr_lag:
            P_fcr = x[1]
            # first order filter of FCR power
            P_fcr_dot = (-K*Δf - P_fcr)/T_fcr
        else:
            P_fcr = -K*Δf
    
        P_load = 0
        if t>=0 and t<T_ΔP:
            P_load = ΔP_load 
        
        # Swing equation:
        ΔP = -P_load + P_fcr # net excess of mechanical power
        Ω_dot = ΔP/(J*Ω)
        f_dot = Ω_dot/(2*pi)

        if fcr_lag:
            x_dot = (f_dot, P_fcr_dot)
        else:
            x_dot = (f_dot,)
        
        return x_dot

    # Initial conditions
    f_ini = 50 # Hz
    if fcr_lag:
        P_fcr_ini = 0 # MW
        x_ini = (f_ini, P_fcr_ini)
    else:
        x_ini = (f_ini,)

    # Solve ODE:
    sol = solve_ivp(fder, (t_ini, t_fin), x_ini,
                    method='BDF', max_step=(t_fin-t_ini)/100, rtol=1e-4)
    # Extract simulation output
    f = sol.y[0]
    if fcr_lag:
        P_fcr =  sol.y[1]
    else:
        P_fcr = -K*(f-f0)
    return sol.t, f, P_fcr, f_lim

### Helper
def find_nadir(f):
    """find the frequency nadir in the vector `f`
    Returns (i_nadir, f_nadir) which are (None, None) if no nadir is found
    """
    i_min = np.argmin(f)

    if i_min < (len(f)-1) and i_min>0 and f[i_min]<=f[-1]-0.001:
        i_nadir = i_min
        f_nadir = f[i_min]
    else:
        i_nadir = None
        f_nadir = None
    
    return i_nadir, f_nadir

### Plot function
def plot_freq_response(t, f, P_fcr, f_lim,
                       ΔP_load, T_ΔP, H, s, T_fcr, fcr, fcr_lag,
                       f_min=None, P_max=None):
    """Plot frequency response.

    Parameters:
    - Simulation solution: t, f, P_fcr, f_lim (output of `sim_freq_response`)
    - Simulation parameters: ΔP_load, T_ΔP, H, s, T_fcr, fcr, fcr_lag (params of `sim_freq_response`)
    - vertical axis limits can be optionally forced with `f_min` and `P_max`.

    Returns fig, (ax1, ax2)
    """
    margin = 0.05 # margins for min and max values of axis limits

    P_load = np.where((t<0) | (t>T_ΔP), 0.0, ΔP_load)
    
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, num=1)
    
    ### 1) Frequency plot ######
    title_f = f'Frequency response (H={H:.0f} s)'
    title_f += ' with FCR' if fcr else ' *without* FCR'
    
    if f_min is None:
        f_min = np.min(f)

    ax1.plot(t, f, '-')
    ax1.axhline(f0, ls=':', color='k')

    # Initial RoCoF linear trend:
    RoCoF = -f0 /(2*H) * ΔP_load
    RoCoF_label = f'RoCoF {RoCoF*1000:.0f} mHz/s'
    ax1.axline((0, f0), slope=RoCoF, ls=':', label=RoCoF_label)

    # Final frequency with FCR alone
    if fcr:
        ax1.plot((0, t_fin),(f_lim, f_lim), 'C0--', alpha=0.5, label=f'f final {f_lim:.2f} Hz')
    
    # Nadir:
    i_nadir, f_nadir = find_nadir(f)
    if i_nadir is not None:
        t_nadir = t[i_nadir]
        ax1.plot(t_nadir, f_nadir, 'D', color='tab:red', label='Nadir')

    # 0 Hz limit
    ax1.axhspan(-10, 0, color='black', alpha=0.7)

    # Blackout
    if f[-1] < f0*1e-3: # ~0 Hz
        ax1.axvspan(t[-1], t_fin, color='tab:red', alpha=0.5, label='blackout')
    
    # Annotations, grid and legend
    f_max = f0 if ΔP_load>=0 else f0+(f0-f_min) # show overspeed
    δf = (f0 - f_min)*margin # freq axis margin
    
    ax1.set(
        title=title_f,
        ylabel='frequency (Hz)', 
        ylim=(f_min-δf, f_max+δf),
        xlim=(t_ini, t_fin)
    )
    ax1.grid()
    ax1.legend(loc='upper right')    
    
    
    ### 2) Power plot ######
    title_P = 'FCR power response'
    if fcr:
        title_P += f' (s={s:.0%})'
        title_P += f' with lag τ={T_fcr:.1f} s' if fcr_lag else ''
    else:
        title_P += ' (inactive)'

    ax2.plot(t, P_load, 'k:', label='load excess')
    ax2.plot(t, P_fcr, 'C2-', label='FCR')
    if fcr_lag:
        ax2.axvspan(0, T_fcr, alpha=0.15, color='tab:red', label='lag τ')
    

    # Annotations, grid and legend
    ax2.set(
        title=title_P,
        xlabel='time (s)',
        ylabel='Power (pu)',
    )
    if P_max is not None:
        δP = P_max*margin
        P_min = 0 if ΔP_load>=0 else -P_max
        ax2.set_ylim(P_min-δP, P_max+δP)
    ax2.grid()
    ax2.legend(loc='upper right') 

    plt.show()
    
    return fig, (ax1, ax2)

### Interactive plot function
def freq_response_interact(ΔP_load=0.1, H=1, s=0.10, T_fcr=1,
                           fcr=False, fcr_lag=False, zoom_freq=False, T_ΔP='permanent'):
    """interactive plot of frequency response, meant to be given to `ipywidgets.interact`
    """
    if T_ΔP=='permanent':
        T_ΔP=t_fin
    elif T_ΔP=='4 s':
        T_ΔP=4
    else:
        ValueError("T_ΔP should be 'permanent' or '4 s'")
    # 1) Simulate
    sim_params = dict(ΔP_load=ΔP_load, T_ΔP=T_ΔP, H=H, s=s, T_fcr=T_fcr,
                      fcr=fcr, fcr_lag=fcr_lag)
    t, f, P_fcr, f_lim = sim_freq_response(**sim_params)

    # 2) Plot
    f_min = 49 if zoom_freq else 0 # Hz
    P_max = 0.14
    
    fig, (ax1, ax2) = plot_freq_response(
        t, f, P_fcr, f_lim,
        **sim_params,
        f_min=f_min, P_max=P_max)