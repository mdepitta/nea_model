import numpy as np
import scipy as sp
import scipy.constants as spc

# Custom modules
import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'),'Ongoing.Projects/pycustommodules'))
import save_utils as svu
import general_utils as gu
import geometry as geom

#-----------------------------------------------------------------------------------------------------------------------
# Brian2 import: we use Brian CPP-standalone code generation for fast parallelized simulations
#-----------------------------------------------------------------------------------------------------------------------
from brian2 import *
code_dir = './codegen'
prefs.GSL.directory = '/usr/include/'   ## The directory where the GSL library headings are found
set_device('cpp_standalone',directory=code_dir,build_on_run=False)
prefs.devices.cpp_standalone.openmp_threads = 2 ## The number of threads used in the parallelization (machine-dependent)
prefs.logging.file_log = False
prefs.logging.delete_log_on_exit = True

import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------------------
## Build User-defined convenience functions to be also called by equations in neuron models by Brian
#-----------------------------------------------------------------------------------------------------------------------
def Hill(x,K,n):
    return x**n/(x**n+K**n)
Hill = Function(Hill,arg_units=[mmolar,mmolar,1], return_unit=1,auto_vectorise=False)
Hill_cpp = '''
    #include <math.h>
    double Hill(double x,double K,double n)
    {
        return pow(x,n)/(pow(x,n)+pow(K,n));
    };
    '''
Hill.implementations.add_implementation('cpp',Hill_cpp,compiler_kwds={'headers': ['"math.h"']})

def ThermalVoltage(T):
    return spc.R*(T+273.15)/spc.physical_constants['Faraday constant'][0]
ThermalPotential = Function(ThermalVoltage,arg_units=[1], return_unit=1,auto_vectorise=False)
ThermalVoltage_cpp = '''
    #include <gsl/gsl_const_mksa.h>
    double ThermalPotential(const double T)
    {
        const double R = GSL_CONST_MKSA_MOLAR_GAS;
        const double F = GSL_CONST_MKSA_FARADAY;
        return R*(T+273.15)/F;
    }
    '''
ThermalPotential.implementations.add_implementation('cpp',ThermalVoltage_cpp,
                                                  compiler_kwds={'headers': ['"gsl_const_mksa.h"'],
                                                                 'include_dirs': ['/usr/include/gsl']})
def NernstPotential(x_e,x_i,z,T):
    """
    Nernst potential in volts (w/out units)

    Input parameters (w/out units):
    - x_e : float   Intracellular concentration
    - x_i : float   Extracellyular concentration
    - z   : int     Ion valence
    - T   : float   Temperature in ^C

    Return:
    - E_x : Nernst Reverse Potential in volt (W/OUT units)
    """
    V_T = ThermalVoltage(T)
    return V_T/z*np.log(x_e/x_i)
NernstPotential = Function(NernstPotential,arg_units=[mmolar,mmolar,1,1], return_unit=1,auto_vectorise=False)
NernstPotential_cpp = '''
    #include <gsl/gsl_const_mksa.h>
    const double R = GSL_CONST_MKSA_MOLAR_GAS; 
    const double F = GSL_CONST_MKSA_FARADAY;
    double ThermalVoltage(const double T)
    {
        return R*(T+273.15)/F;
    };
    double NernstPotential(double x_e,double x_i,double z,const double T)
    {
        return ThermalVoltage(T)*log(x_e/x_i)/z;
    };
    '''
NernstPotential.implementations.add_implementation('cpp',NernstPotential_cpp,
                                                   dependencies={'log': DEFAULT_FUNCTIONS['log']},
                                                   compiler_kwds={'headers': ['"gsl_const_mksa.h"'],
                                                                  'include_dirs': ['/usr/include/gsl']})

def IversonBrackets(x,eps):
    """
    Nernst potential in volts (w/out units)

    Input parameters (w/out units):
    - x_e : float   Intracellular concentration
    - x_i : float   Extracellyular concentration
    - z   : int     Ion valence
    - T   : float   Temperature in ^C

    Return:
    - E_x : Nernst Reverse Potential in volt (W/OUT units)
    """
    return np.abs(x)/(np.abs(x)+eps)
IversonBrackets = Function(IversonBrackets,arg_units=[1,1], return_unit=1,auto_vectorise=False)
IversonBrackets_cpp = '''
    #include <gsl/gsl_const_mksa.h>
    double IversonBrackets(double x, const double eps)
    {
        return abs(x)/(abs(x)+eps);
    };
    '''
IversonBrackets.implementations.add_implementation('cpp',IversonBrackets_cpp,
                                                   dependencies={'abs': DEFAULT_FUNCTIONS['abs']})

def VoltageGradient(v_i,x_i,x_e,z_x,T=37,model='ghk'):
    """
    Compute VoltageGradient: note in reality in the GHK the gradient is in C/dm^3

    Input parameters
    - v_i : Intracellular potential (in mV)
    - x_i : Intracellular concentration (in mM)
    - x_e : Intracellular concentration (in mM)
    - z_x : ion valence (signed)
    - T   :   Temperature (in ^C)
    - model : 'ghk' | 'lin'

    Return
    - dv  : in C/dm^3 (if model=='ghk') or in mV (if model=='lin')
    """
    F = spc.physical_constants['Faraday constant'][0]
    V_T = ThermalVoltage(T)*1e3 # in mV
    if model=='ghk':
        dv = -(z_x**2)*F*v_i/V_T*(x_i-x_e*np.exp(-v_i/V_T/z_x))/np.expm1(-v_i/V_T/z_x)
    elif model=='lin':
        dv = v_i - NernstPotential(x_e,x_i,z_x,T)
    return dv

def current_comparison(x_i,x_e,z_x,mode='norm'):
    """
    Compare behaviour of 'ghk' vs. 'lin'
    """
    npts = 1000
    v = np.linspace(-100,100,npts).astype(float)
    i_ghk = VoltageGradient(v, x_i, x_e, z_x, T=37, model='ghk')
    i_np = VoltageGradient(v, x_i, x_e, z_x, T=37, model='lin')

    if mode=='norm':
        i_ghk = gu.normalize_elements(i_ghk)
        i_np = gu.normalize_elements(i_np)
    elif mode=='normmax':
        i_ghk /= np.abs(np.max(i_ghk))
        i_np /= np.abs(np.max(i_np))

    fig,ax = plt.subplots(1,1)
    ax.plot(v,i_ghk,'k-')
    ax.plot(v,i_np,'r-')

def I_Nat(v,g_Nat,E_Na,U_m,U_h,W_m,W_h):
    a_m = -0.182*(v-U_m)/np.expm1((U_m-v)/W_m)
    b_m=-0.124*(U_m-v)/np.expm1((v-U_m)/W_m)
    a_h=-0.015*(U_h-v)/np.expm1((v-U_h)/W_h)
    b_h=-0.015*(v-U_h)/np.expm1((U_h-v)/W_h)
    m_inf=a_m/(a_m+b_m)
    h_inf=a_h/(a_h+b_h)
    return g_Nat*m_inf**3*h_inf*(v-E_Na)

def I_Kv(v,g_K,E_K,U_n,W_n):
    n_inf = 1/(1+exp((U_n-v)/W_n))
    return g_K*n_inf*(v-E_K)

def I_Cl(v,g_Cl,E_Cl):
    return g_Cl*(v-E_Cl)

def I_NKA(I_max,N_i,N_e,K_e,model=0,v=None,T=37):
    if model==0:
        # Sibille et al. (also Foncelle)
        I = I_max*(Hill(N_i,13,1)**3)*(Hill(K_e,0.2,1)**2)
    elif model==1:
        # Luo et al. 1994
        V_T = ThermalVoltage(T)*1e3  # mV
        s = np.expm1(N_e/67.3)/7
        I = I_max*Hill(N_i,13,1.5)*Hill(K_e,0.2,1)
        I /= (1+0.1245*np.exp(-0.1*v/V_T)+0.0365*s*np.exp(-v/V_T))
    return I

def I_KCC(g_KCC,K_e,K_i,C_e,C_i,T=37):
    E_K = NernstPotential(K_e, K_i, 1,T)*1e3
    E_Cl = NernstPotential(C_e,C_i,-1,T)*1e3
    return g_KCC*(E_K-E_Cl)

def estimate_INKA(I_max,model=0,T=37):
    npts = 100
    v_ = np.linspace(-100,-50,npts)
    N_i = 10
    N_e = 145
    E_Na = NernstPotential(N_e,N_i,1,T)*1e3
    I_Na = I_Nat(v_, 2e6, E_Na, -38, -66, 6, 6)
    K_i = 130
    K_e = 3
    E_K = NernstPotential(K_e, K_i, 1, T)*1e3
    I_K = I_Kv(v_, 1e5, E_K, 18.7, 9.7)
    C_i = 5
    C_e = 130
    E_Cl = NernstPotential(C_e, C_i, -1, T)*1e3
    I_L = I_Cl(v_, 5e2, E_Cl)
    fig,ax = plt.subplots(1,1)
    I_ion = I_Na+I_K+I_L
    ax.plot(v_,I_ion,'k-')
    for Imax in np.linspace(0,1,10)[1:]*I_max:
       ax.plot(v_,I_NKA(Imax, N_i, N_e, K_e, model=model, v=v_, T=T))

def estimate_equilibrium(npts=100,**kwargs):
    v_rest = np.linspace(-90,-40,int(npts))*mV
    pars = lpc5_parameters(model='hh-ecs',**kwargs)
    # Resolve E_K(v,N_i,N_e,K_e)
    N_i = pars['N0_i']/mmolar
    N_e = pars['N0_e']/mmolar
    K_i = pars['K0_i']/mmolar
    V_T = ThermalVoltage(T)*volt  # mV
    s = np.expm1(N_e/67.3)/7
    I_NKP = pars['I_NKA']*Hill(N_i,13,1.5)/(1+0.1245*np.exp(-0.1*v_rest/V_T)+0.0365*s*np.exp(-v_rest/V_T))
    A = -(1/3)*I_Nat(v_rest,pars['g_Na'],pars['E_Na'],pars['U_m'],pars['U_h'],pars['W_m'],pars['W_h'])/I_NKP
    EK_Na = V_T*np.log(A*pars['zeta_K']/(1-A)/pars['K0_i'])

    # Resolve E_K(v)
    e_k = np.linspace(-150,-40,npts)*mV
    VR,EK = np.meshgrid(v_rest,e_k,indexing='ij')
    K_e = K_i*np.exp(EK/V_T)
    F_k = I_Kv(VR,pars['g_K'],EK,pars['U_n'],pars['W_n'])-2*I_NKA(pars['I_NKA'],N_i,N_e,K_e,v=VR/mV,model=1,T=pars['T_exp'])-I_KCC(pars['g_KCC'],K_e,K_i,pars['C0_e'],pars['C0_i'],T=pars['T_exp'])*mV
    sol_K = geom.find_contour(np.asarray(F_k),v_rest/volt,e_k/volt,0)

    # Resolve E_K(v,E_Cl)
    E_K_Cl = (pars['g_KCC']*pars['E_Cl']-I_Cl(v_rest,pars['g_Cl'],pars['E_Cl']))/pars['g_KCC']

    # Plot results
    fig,ax = plt.subplots(1,1)
    ax.plot(v_rest,EK_Na,'k-')
    ax.plot(sol_K[0],sol_K[-1],'b-')
    ax.plot(v_rest,E_K_Cl,'r-')

def lpc5_parameters(model='hh-neuron',**kwargs):
    pars = {## Concentrations to setup reverse potentials
            'N0_i': 10*mmolar,
            'N0_e': 145*mmolar,
            'K0_i': 130*mmolar,
            'K0_e': 3*mmolar,
            'C0_i': 5*mmolar,
            'C0_e': 130*mmolar,
            ## Neuron Parameters and conductances
            'c_m' : 1*ufarad/cm**2,
            'g_Na': 2.04e6*usiemens/cm**2,
            # 'g_K' : 0.638e6*usiemens/cm**2,
            'g_K': 0.693e6*usiemens/cm**2,
            'g_L_Na': 32.7*usiemens/cm**2,
            'g_L_K' : 0.0*usiemens/cm**2,
            'g_L_Cl': 50*usiemens/cm**2,
            'v_thr' : 0*mvolt,
            ## Gating variables
            'U_m' : -38*mvolt,
            'U_h' : -66*mvolt,
            'U_n' : 18.7*mvolt,
            'W_m' : 6*mvolt,
            'W_h' : 6*mvolt,
            'W_n' : 9.7*mvolt,
            ## Temperature
            'T_exp' : 37, ## Body temperature of the animal
            ## External Stimulation
            'I_dc'  : 0*namp/cm**2,
            'I_ramp': 0*namp/cm**2,
            'T_ramp': 100*second,
            }

    if model=='hh-ecs':
        ## Geometry
        # pars['S']       = 150*um**2
        # pars['Lambda']  = 500*um**3
        # pars['Lambda_E']  = 500*um**3
        pars['S']       = 700*um**2
        pars['Lambda']  = 1750*um**3
        pars['Lambda_E']  = 500*um**3

        ## I_NKP
        pars['I_NKA']   = 0*namp/cm**2
        # pars['zeta_Na'] = 30*mmolar ## Can be between 25--30 Clausen et al., Front Physiol. 2017
        # pars['zeta_K']  = 1*mmolar  ## Clausen et al., Front Physiol. 2017
        pars['zeta_Na'] = 13*mmolar ## Can be between 25--30 Clausen et al., Front Physiol. 2017
        pars['zeta_K']  = 0.2*mmolar  ## Clausen et al., Front Physiol. 2017
        ## I_KCC
        pars['g_KCC']   = 0*usiemens/cm**2

    pars = gu.varargin(pars,**kwargs)

    pars['T_adj'] = 2.3**(0.1*(pars['T_exp']-21))
    # if model=='hh-neuron':
    pars['E_Na'] = NernstPotential(pars['N0_e'],pars['N0_i'],1,pars['T_exp'])*volt
    pars['E_K']  = NernstPotential(pars['K0_e'],pars['K0_i'],1,pars['T_exp'])*volt
    pars['E_Cl'] = NernstPotential(pars['C0_e'],pars['C0_i'],-1,pars['T_exp'])*volt

    ## Physics constants
    pars['F'] = spc.physical_constants['Faraday constant'][0]*coulomb/mole

    return pars

def lpc5_neuron(N,params,model='hh',name='hh*',dt=None):
    eqs = Equations('''
    a_m=-0.182/second/mvolt*(v-U_m)/expm1((U_m-v)/W_m)   : hertz
    b_m=-0.124/second/mvolt*(U_m-v)/expm1((v-U_m)/W_m)   : hertz
    a_h=-0.015/second/mvolt*(U_h-v)/expm1((v-U_h)/W_h)   : hertz
    b_h=-0.015/second/mvolt*(v-U_h)/expm1((U_h-v)/W_h)   : hertz
    m_inf=a_m/(a_m+b_m)                                : 1
    h_inf=a_h/(a_h+b_h)                                : 1
    n_inf=1/(1+exp(-(v-U_n)/W_n))                      : 1
    tau_m=1e-3/(a_m+b_m)/T_adj                         : second
    tau_h=1e-3/(a_h+b_h)/T_adj                         : second
    tau_n=4*ms/(1+exp(-(v+56.56*mvolt)/(44.14*mvolt)))/T_adj      : second
    I_inj=I_dc+I_ramp*t/T_ramp                      : amp/meter**2
    dm/dt=(m_inf-m)/tau_m                           : 1
    dh/dt=(h_inf-h)/tau_h                           : 1
    dn/dt=(n_inf-n)/tau_n                           : 1
    ''')

    if model=='hh':
        eqs += Equations('''
            I_Na=g_Na*m**3*h*(v-E_Na)                       : amp/meter**2
            I_K=g_K*n*(v-E_K)                               : amp/meter**2
            I_Cl=g_Cl*(v-E_Cl)                              : amp/meter**2
            dv/dt=(I_inj-I_Na-I_K-I_Cl)/c_m                 : volt   
        ''')
    else:
        ## w/ ECS
        eqs += Equations('''
                # Compute Intracellular Concentrations
                N_i=clip(N0_i+n_Na/Lambda,0*mmolar,inf*mmolar)  : mmolar
                K_i=clip(K0_i+n_K/Lambda,0*mmolar,inf*mmolar)   : mmolar
                C_i=clip(C0_i+n_Cl/Lambda,0*mmolar,inf*mmolar)  : mmolar
                # Resolve Nernst Potentials
                V_T=ThermalPotential(T_exp)*volt                : volt
                E_Na=NernstPotential(N_e,N_i,1,T_exp)*volt      : volt 
                E_K=NernstPotential(K_e,K_i,1,T_exp)*volt       : volt 
                E_Cl=NernstPotential(C_e,C_i,-1,T_exp)*volt     : volt 
                # Compute Individual fluxes
                I_Na=g_Na*m**3*h*(v-E_Na)                       : amp/meter**2
                I_K=g_K*n*(v-E_K)                               : amp/meter**2
                # Leakage components
                I_L_Na=g_L_Na*(v-E_Na)                          : amp/meter**2
                I_L_K=g_L_K*(v-E_K)                             : amp/meter**2
                I_L_Cl=g_L_Cl*(v-E_Cl)                          : amp/meter**2
                I_L=I_L_Na+I_L_K+I_L_Cl                         : amp/meter**2
                # Transport mechanisms
                I_NKP=I_NKA*Hill(N_i,zeta_Na,1.5)*Hill(K_e,zeta_K,1)/(1+0.1245*exp(-0.1*v/V_T)-0.0052*exp(-v/V_T)*(1-exp(N_e/67.3/mmolar))) : amp/meter**2
                I_KCC=g_KCC*(E_K-E_Cl)                          : amp/meter**2
                # ODEs
                dv/dt=(I_inj-I_Na-I_K-I_L-I_NKP)/c_m             : volt
                dn_Na/dt=-S*(I_Na+I_L_Na+3*I_NKP)/F              : mole
                dn_K/dt=-S*(I_K+I_L_K-2*I_NKP-I_KCC)/F           : mole
                dn_Cl/dt=S*(I_L_Cl+I_KCC)/F                      : mole
                # External variables
                # N_e    : mmolar
                # K_e    : mmolar
                # C_e    : mmolar
                N_e=clip(N0_e-n_Na/Lambda_E,0*mmolar,inf*mmolar)    : mmolar
                K_e=clip(K0_e-n_K/Lambda_E,0*mmolar,inf*mmolar)     : mmolar
                C_e=clip(C0_e-n_Cl/Lambda_E,0*mmolar,inf*mmolar)    : mmolar  
            ''')

    neurons = NeuronGroup(N,eqs,
                         threshold='v>v_thr',
                         reset='',
                         namespace=params,
                         name=name,
                         method='rk4',
                         dt=dt)

    neurons.m = 0.01
    # neurons.h = 0.99
    neurons.h = 0.66
    neurons.n = 0.01

    if model=='hh-ecs':
        neurons.n_Na = 0*mole
        neurons.n_K = 0*mole
        neurons.n_Cl = 0*mole

    return neurons

def astrocyte_parameters(model='astro-ecs',**kwargs):
    # TODO: Specify I_GABA parameters
    pars = {## Concentrations to setup reverse potentials
            'N0_i': 15*mmolar,
            'N0_e': 145*mmolar,
            'K0_i': 100*mmolar,
            'K0_e': 3*mmolar,
            'C0_i': 40*mmolar,
            'C0_e': 130*mmolar,
            'H0_i': 60*nmolar,
            'H0_e': 40*nmolar,
            'G0_i': 25*nmolar,
            'G0_e': 25*nmolar,
            'HBC0_e': 10*mmolar,
            'HBC0_i': 10*mmolar,
            ## Astrocyte Parameters
            'c_m' : 1*ufarad/cm**2,
            ## Temperature
            'T_exp' : 37, ## Body temperature of the animal
            ## External Stimulation
            'I_dc'  : 0*namp/cm**2,
            'I_ramp': 0*namp/cm**2,
            'T_ramp': 100*second,
            ## Geometry
            'S'     : 850*um**2,
            'Lambda': 1000*um**3,
            'Lambda_E': 500*um**3,
            ## Kir
            'g_Kir'   : 175*usiemens/cm**2,
            'zeta_Kir': 13*mmolar,
            ## EAAT
            'sigma_EAAT': 100/um**2,
            'g_EAAT'    : 0*usiemens/cm**2,
            'g_Cl'      : 0*usiemens/cm**2,
            'Omega_Glu' : 25/second,
            ## NKCC
            'g_NKCC'  : 0*usiemens/cm**2,
            ## NKP
            'I_NKA'   : 0*namp/cm**2,
            'zeta_Na' : 10*mmolar,
            'zeta_K'  : 3*mmolar,
            ## GABA
            'g_GABA'  : 0*usiemens/cm**2,
            'P_Cl'    : 1.0,
            'P_HBC'   : 0.66
    }

    pars = gu.varargin(pars,**kwargs)

    pars['T_adj'] = 2.3**(0.1*(pars['T_exp']-21))
    pars['E_H']  = NernstPotential(pars['H0_e'],pars['H0_i'],1,pars['T_exp'])*volt
    pars['E_HBC'] = NernstPotential(pars['HBC0_e'],pars['HBC0_i'],1,pars['T_exp'])*volt
    pars['E_GABA'] = ThermalVoltage(pars['T_exp'])*log((pars['P_Cl']*pars['C0_i']+pars['P_HBC']*pars['HBC0_i'])/(pars['P_Cl']*pars['C0_e']+pars['P_HBC']*pars['HBC0_e']))*volt
    pars['f_Cl'] = pars['P_Cl']/(pars['P_Cl']+pars['P_HBC'])

    pars['N_EAAT'] = pars['sigma_EAAT']*pars['S']
    pars['g_EAAT'] *= pars['N_EAAT']
    pars['g_Cl'] *= pars['N_EAAT']

    ## Physics constants
    pars['F'] = spc.physical_constants['Faraday constant'][0]*coulomb/mole

    return pars

def astrocyte_cell(N,params,model='astro',name='astro*',dt=None):
    # TODO: E_Glu becomes dynamically dependent on G_e = G_0e
    eqs = Equations('''
        N_i=clip(N0_i+n_Na/Lambda,0*mmolar,inf*mmolar)  : mmolar
        K_i=clip(K0_i+n_K/Lambda,0*mmolar,inf*mmolar)   : mmolar
        C_i=clip(C0_i+n_Cl/Lambda,0*mmolar,inf*mmolar)  : mmolar
        G_i=clip(G0_i+n_Glu/Lambda,0*mmolar,inf*mmolar) : mmolar
        # Define relevant quantities
        V_T=ThermalPotential(T_exp)*volt                : volt
        E_Na=NernstPotential(N_e,N_i,1,T_exp)*volt      : volt 
        E_K=NernstPotential(K_e,K_i,1,T_exp)*volt       : volt 
        E_Cl=NernstPotential(C_e,C_i,-1,T_exp)*volt     : volt
        E_Glu=NernstPotential(G_e,G_i,-1,T_exp)*volt    : volt
        ## E_EAAT=(3*E_Na+E_H+2*E_Cl-E_K-E_Glu)/4       : volt
        E_EAAT=(3*E_Na+E_H-E_K-E_Glu)/2                 : volt
        E_GABA=V_T*log((P_Cl*C_i+P_HBC*HBC0_i)/(P_Cl*C_e+P_HBC*HBC0_e)) : volt
        # Define currents
        I_Kir=g_Kir*(v-E_K)/(2+exp(1.62*(v-E_K)/V_T))*Hill(K_e,zeta_Kir,1)   : amp/meter**2
        I_NKP=I_NKA*Hill(N_i,zeta_Na,1.5)*Hill(K_e,zeta_K,1)*(1+0.1245*exp(-0.1*v/V_T)-0.0052*exp(-v/V_T)*(1-exp(N_e/67.3/mmolar))) : amp/meter**2
        I_NKCC=g_NKCC*(E_Na+E_K-2*E_Cl)                 : amp/meter**2
        I_GluT=-g_EAAT*(v-E_EAAT)                       : amp/meter**2
        ## I_GluT=g_EAAT*(v-E_EAAT)                        : amp/meter**2
        I_Cl=g_Cl*(v-E_Cl)*IversonBrackets((v-E_EAAT)/volt,1e-9) : amp/meter**2
        I_GABA=g_GABA*(v-E_GABA)                        : amp/meter**2        
        # r.h.s.    
        dv/dt=(-I_GABA-I_Kir-I_NKP-I_GluT-I_Cl)/c_m     : volt
        dn_Na/dt=-S*(3*I_NKP-3*I_GluT-I_NKCC)/F         : mole
        dn_K/dt=-S*(I_Kir+I_GluT-I_NKCC-2*I_NKP)/F      : mole
        ## dn_Cl/dt=S*(f_Cl*I_GABA+2*I_NKCC-2*I_GluT)/F : mole
        dn_Cl/dt=S*(f_Cl*I_GABA+2*I_NKCC+I_Cl)/F        : mole
        dn_Glu/dt=S*I_GluT/F - Omega_Glu*n_Glu          : mole
        # External variables
        # N_e    : mmolar
        # K_e    : mmolar
        # C_e    : mmolar
        # G_e    : mmolar
        N_e=clip(N0_e-n_Na/Lambda_E,0*mmolar,inf*mmolar)    : mmolar
        K_e=clip(K0_e-n_K/Lambda_E,0*mmolar,inf*mmolar)     : mmolar
        C_e=clip(C0_e-n_Cl/Lambda_E,0*mmolar,inf*mmolar)    : mmolar
        G_e=clip(G0_e-n_Glu/Lambda_E,0*mmolar,inf*mmolar)   : mmolar
    ''')

    astrocytes = NeuronGroup(N,eqs,
                         namespace=params,
                         name=name,
                         method='rk4',
                         dt=dt)

    astrocytes.n_Na = 0*mole
    astrocytes.n_K = 0*mole
    astrocytes.n_Cl = 0*mole
    astrocytes.n_Glu = 0*mole

    return astrocytes


def lpc5_simulation(duration=1,model='hh-neuron',**kwargs):
    # start_scope()
    device.delete(force=True)
    ## Build the neuron model and monitors
    params = lpc5_parameters(model=model,T_ramp=duration*second,**kwargs)
    # print(params['E_Na'])
    # print(params['E_K'])
    # print(params['E_Cl'])

    # Initialize model
    cell = lpc5_neuron(1,params,model=model,name='HH',dt=0.1*us)
    # cell.v = NernstPotential(params['C0_e'],params['C0_i'],-1,params['T_exp'])*volt
    cell.v = -70*mV
    # if model=='hh-ecs':
    #     cell.N_e = params['N0_e']
    #     cell.K_e = params['K0_e']
    #     cell.C_e = params['C0_e']

    # Set monitors
    if model=='hh-neuron':
        variables = ['v']
    elif model=='hh-ecs':
        variables = ['v','E_Cl','E_K','E_Na','I_L_Na','I_Na','I_K','I_L_Cl','I_KCC','I_NKP'] # ,'C_i','N_i','K_i','I_K','I_Na'
    sv_mon = StateMonitor(cell,variables=variables,record=True,dt=0.1*ms,name='svmon')
    # Gene
    network = Network([cell,sv_mon])
    ## Run the simulator
    network.run(duration=duration*second,report='text')
    device.build(directory=code_dir, clean=True)

    vfactor = mV
    ifactor = nA/cm**2

    ## Visualizing data
    if model=='hh-neuron':
        fig, ax = plt.subplots(1, 1)
        ax.plot(sv_mon.t, sv_mon.v[:].T, 'k-')
    elif model=='hh-ecs':
        fig, ax = plt.subplots(6, 1,sharex=True)
        ax[0].plot(sv_mon.t, sv_mon.v[:].T/vfactor, 'k-')
        ax[0].plot(sv_mon.t, sv_mon.E_Na[:].T/vfactor, 'g-')
        ax[0].plot(sv_mon.t, sv_mon.E_K[:].T/vfactor, 'b-')
        ax[0].plot(sv_mon.t, sv_mon.E_Cl[:].T/vfactor, 'y-')
        ##
        ax[1].plot(sv_mon.t,sv_mon.I_Na[:].T/ifactor,'k-')
        ax[1].plot(sv_mon.t,sv_mon.I_L_Na[:].T/ifactor,'r-')
        ##
        ax[2].plot(sv_mon.t,sv_mon.I_K[:].T/ifactor,'m-')
        ##
        ax[3].plot(sv_mon.t,sv_mon.I_L_Cl[:].T/ifactor,'g-')
        ##
        ax[4].plot(sv_mon.t,sv_mon.I_KCC[:].T/ifactor,'c-')
        ##
        ax[5].plot(sv_mon.t,sv_mon.I_NKP[:].T/ifactor,'c-')

        # ax[1].plot(sv_mon.t, sv_mon.C_i[:].T, 'm-')
        # ax[1].plot(sv_mon.t, sv_mon.C_i[:].T, 'r-')
        # ax[1].plot(sv_mon.t, sv_mon.N_i[:].T, 'r-')
        # ax[1].plot(sv_mon.t,sv_mon.I_Na[:].T,'r-')
        # ax[2].plot(sv_mon.t,sv_mon.I_K[:].T,'b-')
        # ax[3].plot(sv_mon.t,sv_mon.I_Cl[:].T,'g-')
        # ax[4].plot(sv_mon.t,sv_mon.I_NKP[:].T,'y-')
        # ax[5].plot(sv_mon.t,sv_mon.I_KCC[:].T,'m-')
        # ax[1].plot(sv_mon.t, sv_mon.n_Cl[:].T, 'm-')
        # ax[1].hlines(0,*sv_mon.t[[0,-1]])
        # ax[2].plot(sv_mon.t, sv_mon.I_Cl[:].T, 'r-')
        # ax[3].plot(sv_mon.t, sv_mon.n_K[:].T, 'g-')

    (device.delete(force=True))

def astrocyte_simulation(duration=1,model='astro-ecs',**kwargs):
    device.delete(force=True)
    ## Build the neuron model and monitors
    params = astrocyte_parameters(model=model, **kwargs)
    # print(params['E_H'])
    # print(params['E_HBC'])
    # print(params['E_GABA'])
    print(params['N_EAAT'])
    print(params['g_EAAT'])
    print(params['g_Cl'])

    # Initialize model
    cell = astrocyte_cell(1, params, model=model, name='AC', dt=0.1*us)
    # cell.v = NernstPotential(params['C0_e'], params['C0_i'], -1, params['T_exp'])*volt
    cell.v = -90*mV

    variables = ['v', 'C_i', 'N_i', 'K_i', 'G_i', 'E_Na','E_Cl', 'E_K', 'E_EAAT']
    sv_mon = StateMonitor(cell, variables=variables, record=True, dt=0.1*ms, name='svmon')
    # Gene
    network = Network([cell, sv_mon])
    ## Run the simulator
    network.run(duration=duration*second, report='text')
    device.build(directory=code_dir, clean=True)

    fig, ax = plt.subplots(6, 1, sharex=True)
    ax[0].plot(sv_mon.t, sv_mon.v[:].T, 'k-')
    ax[0].plot(sv_mon.t, sv_mon.E_Cl[:].T, 'y-')
    ax[1].plot(sv_mon.t, sv_mon.N_i[:].T, 'g-')
    ax[2].plot(sv_mon.t, sv_mon.K_i[:].T, 'b-')
    ax[3].plot(sv_mon.t, sv_mon.C_i[:].T, 'y-')
    ax[4].plot(sv_mon.t, sv_mon.G_i[:].T, 'm-')
    ax[5].plot(sv_mon.t, sv_mon.E_EAAT[:].T, 'c-')

    device.delete(force=True)


def explore_eaat(g=1):

    g *= 1e2*nS/cm**2

    ## Concentrations
    N0_e = 145*mmolar
    N0_i = 10*mmolar
    K0_e = 3*mmolar
    K0_i = 130*mmolar
    H0_e = 10**-7.2*mmolar
    H0_i = 10**-7.4*mmolar

    ## Geometry factors
    Lambda_e = 525*umetre**3
    S_a = 25e3*umetre**2
    tau = 50*ms
    F = spc.physical_constants['Faraday constant'][0]*coulomb/mole
    eta = F*Lambda_e/S_a/tau

    # print(eta*1*mmolar/g)

    ## Nernst Potentials
    E_Na = NernstPotential(N0_e,N0_i,1,T)*mV
    E_K = NernstPotential(K0_e,K0_i,1,T)*mV
    E_H = NernstPotential(H0_e,H0_i,1,T)*mV

    rhs = lambda g_T,G_e,G_i : eta/g_T*G_e - NernstPotential(G_e,G_i,-1,T)*mV
    g_e = np.logspace(-200,0,50)
    g_i = np.logspace(1,120,50)
    Ge,Gi = np.meshgrid(g_e,g_i,indexing='ij')*mmolar

    vr = rhs(g,Ge,Gi)
    print(vr.min())
    # Find the index (flattened)
    flat_index = np.argmin(rhs)
    # Convert to row, column indices
    row,col = np.unravel_index(flat_index,vr.shape)
    G0_e = Ge[row,col]
    G0_i = Gi[row,col]

    E_Glu = NernstPotential(G0_e,G0_i,-1,T)*mV
    print('E_AAT\t',0.5*(3*E_Na+E_H-E_K-E_Glu))

    # fig,ax = plt.subplots(1,1)
    # cb = ax.pcolormesh(vr,cmap='Reds_r')
    # plt.colorbar(cb)

if __name__=="__main__":
    # #-------------------------------------------------------------------------------------------------------------------
    # # Verify Reverse Potentials
    # #-------------------------------------------------------------------------------------------------------------------
    T= 37
    # print('E_Na\t',NernstPotential(126,9,1,T))
    # print('E_K\t',NernstPotential(2.5,130,1,T))
    # print('E_Cl\t',NernstPotential(130,10,-1,T))
    # print('E_HBC\t',NernstPotential(15,8,-1,T))
    # print('E_GABA\t',-(NernstPotential(15,8,-1,T)+NernstPotential(130,10,-1,T))/2)

    # # #-------------------------------------------------------------------------------------------------------------------
    # # # Testing reverse potentials
    # # #-------------------------------------------------------------------------------------------------------------------
    # print('E_AMPA',ThermalVoltage(T)*np.log((145+1.2*3)/(10+1.2*130)))
    # print('E_GABA',-ThermalVoltage(T)*np.log((130+0.4*35)/(5+0.4*20)))
    print('E_GABA_astro',-ThermalVoltage(T)*np.log((130+0.4*35)/(50+0.4*10)))

    # #-------------------------------------------------------------------------------------------------------------------
    # # Exploring reverse potentials
    # #-------------------------------------------------------------------------------------------------------------------
    # explore_eaat(g=1000)

    # # #-------------------------------------------------------------------------------------------------------------------
    # # # Comparison of GHK vs. NP formalisms
    # # #-------------------------------------------------------------------------------------------------------------------
    # N_i = 10
    # N_e = 145
    # current_comparison(N_i, N_e, 1,mode='norm')
    # K_i = 130
    # K_e = 3
    # current_comparison(K_i, K_e, 1,mode='norm')
    # C_i = 5
    # C_e = 130
    # current_comparison(C_i, C_e, -1,mode='norm')

    # # #-------------------------------------------------------------------------------------------------------------------
    # # # Estimation of I_NKP
    # # #-------------------------------------------------------------------------------------------------------------------
    # I_max = 1e5 ## This suggests that a reasonable I_NKP to get a resting potential around -80 -- -70 mV is I_max~1e4 -- 2.5e5 nA/cm2
    # estimate_INKA(I_max,model=1,T=37)

    # #-------------------------------------------------------------------------------------------------------------------
    # # Generate Neuron Simulation
    # #-------------------------------------------------------------------------------------------------------------------
    # print(lpc5_parameters(I_dc=1e4.5*nA/cm**2))
    ## Single Neuron
    # lpc5_simulation(duration=1,model='hh',I_dc=10**4.2*nA/cm**2)
    ## Single Neuron

    ##------------------------------------------------------------------------
    ## Simulation /2: Effect of NKP in prolonging spiking
    ##------------------------------------------------------------------------
    # lpc5_simulation(duration=1, model='hh-ecs',
    #                 I_dc=10**4.5*nA/cm**2,
    #                 I_NKA=0**6*nA/cm**2)
    # lpc5_simulation(duration=1, model='hh-ecs',
    #                 I_dc=10**4.5*nA/cm**2,
    #                 I_NKA=10**6*nA/cm**2)

    ##------------------------------------------------------------------------
    ## Analysis /2: Neuron Equilibrium for IKCC and INKA
    ##------------------------------------------------------------------------
    # estimate_equilibrium(100,
    #                      N0_e=120*mmolar,
    #                      K0_i=120*mmolar,
    #                      C0_i=10*mmolar,
    #                      g_KCC =0.22e3*uS/cm**2,
    #                      I_NKA=1e4*nA/cm**2)

    # lpc5_simulation(duration=10, model='hh-ecs',
    #                 I_dc=0**4.3*nA/cm**2,
    #                 C0_i=9.47*mmolar,
    #                 g_KCC=1e4*uS/cm**2,
    #                 I_NKA=1e3*nA/cm**2)

    # # Michelangelo's last parameter set
    # lpc5_simulation(duration=1, model='hh-ecs',
    #                 I_dc=0**4.3*nA/cm**2,
    #                 g_L_Na=30*uS/cm**2,
    #                 g_KCC=60*uS/cm**2,
    #                 I_NKA=7e3*nA/cm**2)

    # ##------------------------------------------------------------------------
    # ## Simulation /3: Effect of NKP in prolonging spiking
    # ##------------------------------------------------------------------------
    # lpc5_simulation(duration=1, model='hh-ecs',
    #                 I_dc=0**4.3*nA/cm**2,
    #                 C0_i=9.47*mmolar,
    #                 g_KCC=0.*uS/cm**2,
    #                 I_NKA=0.*nA/cm**2)
    # lpc5_simulation(duration=1, model='hh-ecs',
    #                 I_dc=0**4.3*nA/cm**2,
    #                 N0_e=120*mmolar,
    #                 K0_i=120*mmolar,
    #                 C0_i=10*mmolar,
    #                 g_KCC=0.22e3*uS/cm**2,
    #                 I_NKA=1e4*nA/cm**2)

    # ##------------------------------------------------------------------------
    # ## Simulation /4: Astrocyte tuning
    # ##------------------------------------------------------------------------
    # astrocyte_simulation(duration=1, model='astro-ecs',
    #                      g_Kir=0.*usiemens/cm**2,
    #                      I_NKA=0.*nA/cm**2,
    #                      g_NKCC=0*usiemens/cm**2,
    #                      g_GABA=0*usiemens/cm**2,
    #                      K0_e=3*umolar,
    #                      G0_e=100*umolar,
    #                      Omega_Glu=0/second,
    #                      g_EAAT=1e-1*psiemens/cm**2,
    #                      g_Cl=0*psiemens/cm**2)

    # #-------------------------------------------------------------------------------------------------------------------
    ## Show figures
    # #-------------------------------------------------------------------------------------------------------------------
    plt.show()