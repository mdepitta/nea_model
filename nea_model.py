# Import necessary modules
import matplotlib.pyplot as plt
import scipy.constants as spc
from brian2 import *
from brian2.units.constants import faraday_constant as F
from brian2.units.constants import avogadro_constant as N_A

# Import Warnings to avoid unnecessary warnings
import warnings as wrn
wrn.filterwarnings("ignore")
BrianLogger.suppress_name('resolution_conflict')

#-----------------------------------------------------------------------------------------------------------------------
## General-Purpose Utilities
#-----------------------------------------------------------------------------------------------------------------------
def varargin(pars, **kwargs):
    """
    varargin-like option for user-defined parameters in any function/module
    Use:
    pars = varargin(pars,**kwargs)

    Input:
    - pars     : the dictionary of parameters of the calling function
    - **kwargs : a dictionary of user-defined parameters

    Output:
    - pars     : modified dictionary of parameters to be used inside the calling
                 (parent) function

    Maurizio De Pitta', The University of Chicago, August 27th, 2014.
    """
    for key, val in kwargs.items():
        if key in pars:
            pars[key] = val
    return pars

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

def ThermalPotential(T):
    return spc.R*(T+273.15)/spc.physical_constants['Faraday constant'][0]
ThermalPotential = Function(ThermalPotential,arg_units=[1], return_unit=1,auto_vectorise=False)
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
    # V_T = ThermalVoltage(T)
    V_T = ThermalPotential(T)
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
        double z, e;
        /* Handle degenerate epsilon */
        if (eps <= 0.0) {
            return (x >= 0.0) ? 1.0 : 0.0;
        }
    
        z = x / eps;
        /* Numerically stable sigmoid */
        if (z >= 0.0) {
            e = exp(-z);
            return 1.0 / (1.0 + e);
        } else {
            e = exp(z);
            return e / (1.0 + e);
        }
    };
    '''
IversonBrackets.implementations.add_implementation('cpp',IversonBrackets_cpp,
                                                   dependencies={'abs': DEFAULT_FUNCTIONS['abs']})

def DiffusionFlux(delta_c,c_theta,omega_c,D_coeff):
    """
    Simmetric diffusion flux with saturation (from Lallouette et al., Comput. Gliosci. 2019).

    Input parameters (w/out units):
    - delta_c   : float    Concentration gradient between compartment
    - c_theta   : float    Concentration threshold for diffusion
    - omega_c   : float    Scaling factor
    - D_coeff   : float    Max diffusion rate

    Return:
    - J_diff    : Diffusion Flux
    """
    return -D_coeff/2 * (1 + np.tanh((np.abs(delta_c) - c_theta)/omega_c))*np.sign(delta_c)
DiffusionFlux = Function(DiffusionFlux,arg_units=[mmolar,mmolar,mmolar,mmolar/second], return_unit=mole/second,auto_vectorise=False)
DiffusionFlux_cpp = '''
    #include <math.h>
    int sgn(double val) {
        return (0 < val) - (val < 0);
    };
    double DiffusionFlux(double delta_c, double c_theta, double omega_c, double D_coeff)
    {
        return -D_coeff/2 * (1 + tanh((abs(delta_c) - c_theta)/omega_c))*sgn(delta_c);
    };
    '''
DiffusionFlux.implementations.add_implementation('cpp',DiffusionFlux_cpp,
                                                   dependencies={'abs': DEFAULT_FUNCTIONS['abs'],
                                                                 'tanh': DEFAULT_FUNCTIONS['tanh']})
# -----------------------------------------------------------------------------------------------------------------------
# Generate Model Parameters
# -----------------------------------------------------------------------------------------------------------------------
def nea_parameters(**kwargs):
    pars_neu = {## Concentrations to setup reverse potentials
            'N0_i'   : 10*mmolar,
            'K0_i'   : 130*mmolar,
            'C0_i'   : 5*mmolar,
            'HBC0_i' : 15*mmolar,  # The HBC internally must be such that v_rest > E_GABA (so GABA is always inhibitory for the neuron)
            ## Neuron Parameters and conductances
            'c_m'    : 1*ufarad/cm**2,
            'g_Na'   : 2.04e6*usiemens/cm**2,
            'g_K'    : 0.693e6*usiemens/cm**2,
            'g_L_Na' : 32.7*usiemens/cm**2,
            'g_L_K'  : 0.0*usiemens/cm**2,
            'g_L_Cl' : 50*usiemens/cm**2,
            'v_thr'  : 0*mvolt,
            ## Gating variables
            'U_m'    : -38*mvolt,
            'U_h'    : -66*mvolt,
            'U_n'    : 18.7*mvolt,
            'W_m'    : 6*mvolt,
            'W_h'    : 6*mvolt,
            'W_n'    : 9.7*mvolt,
            ## Temperature
            'T_exp'  : 37, ## Body temperature of the animal
            ## External Stimulation
            'I_dc'   : 0*namp/cm**2,
            'I_ramp' : 0*namp/cm**2,
            'T_ramp' : 100*second,
            ## Geometry
            'S_n'      : 700*um**2,
            'Lambda_n' : 1750*um**3,
            ## Transport
            'I_NKA'  : 0*namp/cm**2,
            'zeta_Na': 13*mmolar,  ## Can be between 25--30 Clausen et al., Front Physiol. 2017
            'zeta_K' : 0.2*mmolar,  ## Clausen et al., Front Physiol. 2017
            ## I_KCC
            'g_KCC'  : 0*usiemens/cm**2,
            ## Permeability Ratios
            'P_K_Na' : 1.2,
            'P_HBC_Cl' : 0.4,
            }

    pars_ecs = {# Extracellular concentrations
            'N0_e'   : 145*mmolar,
            'K0_e'   : 3*mmolar,
            'C0_e'   : 130*mmolar,
            'HBC0_e'  : 35*mmolar,
            'H0_e'    : 50*nmolar,
            'G0_e'    : 25*nmolar,
            'GABA0_e' : 50*nmolar,
            # Diffusion Rates
            'D_Na_e' : 2/second,
            'D_K_e'  : 2/second,
            'D_Cl_e' : 2/second,
            'D_Glu_e'  : 5/second,
            'D_GABA_e' : 5/second,
            'D_Glu'  : 0.33*um**2/msecond,
            'D_GABA' : 0.33*um**2/msecond,
            # Geometry
            'Lambda_e' : 500*um**3,
            # Synaptic geometry
            'l_diff' : 0.13*um,
            't_cleft': 0.2*um
    }

    pars_astro = pars = {## Concentrations to setup reverse potentials
            'N0_a'    : 15*mmolar,
            'K0_a'    : 100*mmolar,
            'C0_a'    : 40*mmolar,
            'H0_a'    : 60*nmolar,
            'G0_a'    : 25*nmolar, ## Glutamate
            'GABA0_a' : 20*mmolar, ## GABA ()
            'HBC0_a'  : 10*mmolar,
            ## Astrocyte Parameters
            'c_m_a'   : 1*ufarad/cm**2,
            ## Temperature
            'T_exp'   : 37, ## Body temperature of the animal
            ## External Stimulation
            'I_dc'    : 0*namp/cm**2,
            'I_ramp'  : 0*namp/cm**2,
            'T_ramp'  : 100*second,
            ## Geometry
            'S_a'       : 850*um**2,
            'Lambda_a'  : 1000*um**3,
            ## Kir
            'g_Kir'     : 175*usiemens/cm**2,
            'zeta_Kir'  : 13*mmolar,
            ## EAAT
            'sigma_EAAT': 100/um**2,
            'g_EAAT'    : 100*usiemens/cm**2,
            'g_T_Cl_a'  : 100*usiemens/cm**2,
            'g_L_Cl_a'  : 60*usiemens/cm**2,
            'Omega_Glu' : 25/second,
            ## GAT
            'g_GAT'   : 6*usiemens/cm**2,
            ## NKCC
            'g_NKCC'  : 0*usiemens/cm**2,
            ## NKP
            'I_NKA_a' : 0*namp/cm**2,
            ## GABA
            'g_GABA'  : 0*usiemens/cm**2,
            'tau_GABA': 50*ms,
            'J_GABA'  : 1/umolar/second,
            ## Buffering
            'D_Na_a'  : 0.1/second,
            'D_K_a'   : 0.1/second,
            'D_Cl_a'  : 0.1/second,
            'D_Na_m'  : 0.01/second,
            'D_K_m'   : 0.1/second,
            'D_Cl_m'  : 0.01/second,
    }

    # Generate default dictionary
    pars = {**pars_neu,**pars_ecs,**pars_astro}

    # Custom-parameters
    pars = varargin(pars,**kwargs)

    # Retrieve partial permeabilities
    pars['pi_AMPA_Na'] = 1.0/(1+pars['P_K_Na'])
    pars['pi_AMPA_K'] = pars['P_K_Na']/(1+pars['P_K_Na'])
    pars['pi_GABA_Cl'] = 1.0/(1+pars['P_HBC_Cl'])

    # Extrapolate useful variables
    pars['T_adj'] = 2.3**(0.1*(pars['T_exp']-21))

    # Extrapolation of the diffusion conductances [volume/sec]
    pars['k_Glu'] = np.pi*pars['D_Glu']*(pars['l_diff']+2*pars['t_cleft'])
    pars['k_GABA'] = np.pi*pars['D_GABA']*(pars['l_diff']+2*pars['t_cleft'])

    return pars

def synapse_parameters(ttype='glu',**kwargs):
    assert any(ttype==t for t in ['glu','gaba']),"Allowed transmitter types (ttype): 'glu' or 'gaba' only"
    if ttype=='glu':
        pars = {'Nt_rel' : 0.1*mmolar,
                'J'      : 1/umolar/second,
                'tau_r'  : 10*msecond,
                'g'      : 10*nsiemens/cm**2,
                'D_Glu'  : 10/second,
        }
    elif ttype=='gaba':
        pars = {'Nt_rel' : 1.0*mmolar,
                'J'      : 1/umolar/second,
                'tau_r'  : 50*msecond,
                'g'      : 10*nsiemens/cm**2,
                'D_GABA' : 10/second,
                }

    pars['Lambda_s'] = 8.75e-3*um**3
    # The current S_s and sigma_R are temporarily taken from the reference below --> need to be better estimated
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130924
    pars['S_s'] = 200*nmeter
    pars['sigma_R'] = 1000*um**-2
    # Estimate receptor density
    pars['R_T'] = pars['sigma_R']/N_A/pars['Lambda_s']*pars['S_s']**2

    pars = varargin(pars,**kwargs)

    return pars

# -----------------------------------------------------------------------------------------------------------------------
# Dummy cell stimulation models
# -----------------------------------------------------------------------------------------------------------------------
def periodic_nodes(N,rates,name='period*',dt=None):
    if isinstance(rates,input.timedarray.TimedArray):
        eqs = Equations('dv/dt = stimulus(t) : 1')
        nmspace = {'stimulus': rates}
    else:
        eqs = Equations('''
                        rate : Hz
                        dv/dt = rate : 1
                        ''')
        nmspace = None
    cells = NeuronGroup(N,eqs,
        threshold='v>=1.0',
        reset='v=0.0',
        name=name,
        namespace=nmspace,
        method='euler',
        dt=dt)
    if not isinstance(rates,str): cells.rate = rates
    cells.v = 0.0
    return cells

@check_units(trp=second)
def poisson_nodes(N,rates,trp=0.0*second,name='poiss*',dt=None):
    # Create equations to also handle case of tme-variant stimulation
    if isinstance(rates,input.timedarray.TimedArray):
        eqs = Equations('rate = stimulus(t) : Hz')
        nmspace = {'stimulus': rates}
    else:
        eqs = Equations('rate : Hz')
        nmspace = None
    cells = NeuronGroup(N,eqs,
        threshold='rand()<rate*dt',
        refractory=trp,
        name=name,
        namespace=nmspace,
        dt=dt)
    if not isinstance(rates,input.timedarray.TimedArray): cells.rate = rates
    return cells

# -----------------------------------------------------------------------------------------------------------------------
# Neuron-ECS-Astrocyte (NEA) Model
# -----------------------------------------------------------------------------------------------------------------------
def nea_node(params,sinput='glu',name='nea*',dt=0.1*us):
    eqs_neuron = '''
        # Hodgkin-Huxley for the neuron model (S1L5)
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
    
        # Compute Intracellular Concentrations
        N_n=clip(N0_i+n_Na_n/Lambda_n,0*mmolar,inf*mmolar)  : mmolar
        K_n=clip(K0_i+n_K_n/Lambda_n,0*mmolar,inf*mmolar)   : mmolar
        C_n=clip(C0_i+n_Cl_n/Lambda_n,0*mmolar,inf*mmolar)  : mmolar
    
        # Resolve Nernst Potentials
        V_T=ThermalPotential(T_exp)*volt                : volt
        E_Na=NernstPotential(N_e,N_n,1,T_exp)*volt      : volt
        E_K=NernstPotential(K_e,K_n,1,T_exp)*volt       : volt
        E_Cl=NernstPotential(C_e,C_n,-1,T_exp)*volt     : volt
    
        # Compute Individual fluxes
        I_Na=g_Na*m**3*h*(v-E_Na)                       : amp/meter**2
        I_K=g_K*n*(v-E_K)                               : amp/meter**2
    
        # Leakage components
        I_L_Na=g_L_Na*(v-E_Na)                          : amp/meter**2
        I_L_K=g_L_K*(v-E_K)                             : amp/meter**2
        I_L_Cl=g_L_Cl*(v-E_Cl)                          : amp/meter**2
        I_L=I_L_Na+I_L_K+I_L_Cl                         : amp/meter**2
    
        # Transport mechanisms
        I_NKP=I_NKA*Hill(N_n,zeta_Na,1.5)*Hill(K_e,zeta_K,1)/(1+0.1245*exp(-0.1*v/V_T)-0.0052*exp(-v/V_T)*(1-exp(N_e/67.3/mmolar))) : amp/meter**2
        I_KCC=g_KCC*(E_K-E_Cl)                          : amp/meter**2
        '''

    if sinput=='glu':
        eqs_neuron += '''
                # Synaptic currents
                E_AMPA=V_T*log((N_e+P_K_Na*K_e)/(N_n+P_K_Na*K_n)) : volt
                I_AMPA=G_AMPA*(v-E_AMPA)  : amp/meter**2
                G_AMPA : siemens/meter**2
    
                # ODEs
                dv/dt=(I_inj-I_AMPA-I_Na-I_K-I_L-I_NKP)/c_m       : volt
                dn_Na_n/dt=-S_n*(-pi_AMPA_Na*I_AMPA+I_Na+I_L_Na+3*I_NKP)/F              : mole
                dn_K_n/dt=-S_n*(-pi_AMPA_K*I_AMPA+I_K+I_L_K-2*I_NKP+I_KCC)/F            : mole
                dn_Cl_n/dt=S_n*(I_L_Cl+I_KCC)/F                       : mole
                '''
    else:
        eqs_neuron += '''
                # Synaptic currents
                E_GABA = -V_T*log((C_e+P_HBC_Cl*HBC0_e)/(C_n+P_HBC_Cl*HBC0_i)): volt
                I_GABA = G_GABA*(v-E_GABA): amp/meter**2
                G_GABA: siemens/meter**2

                # ODEs
                dv/dt = (I_inj-pi_GABA_Cl*I_GABA-I_Na-I_K-I_L-I_NKP)/c_m: volt
                dn_Na_n/dt = -S_n*(I_Na+I_L_Na+3*I_NKP)/F: mole
                dn_K_n/dt = -S_n*(I_K+I_L_K-2*I_NKP+I_KCC)/F: mole
                dn_Cl_n/dt = S_n*(pi_GABA_Cl*I_GABA+I_L_Cl+I_KCC)/F: mole
                '''

    eqs_astro = '''    
        # Concentrations
        N_a=clip(N0_a+n_Na_a/Lambda_a,0*mmolar,inf*mmolar)  : mmolar
        K_a=clip(K0_a+n_K_a/Lambda_a,0*mmolar,inf*mmolar)   : mmolar
        C_a=clip(C0_a+n_Cl_a/Lambda_a,0*mmolar,inf*mmolar)  : mmolar
    
        # Define relevant quantities
        E_Na_a=NernstPotential(N_e,N_a,1,T_exp)*volt      : volt 
        E_K_a=NernstPotential(K_e,K_a,1,T_exp)*volt       : volt 
        E_Cl_a=NernstPotential(C_e,C_a,-1,T_exp)*volt     : volt
        E_Glu_a=NernstPotential(G_e,G0_a,-1,T_exp)*volt   : volt
        E_H_a=NernstPotential(H0_e,H0_a,1,T_exp)*volt     : volt
    
        # Kir + XC currents
        I_Kir=g_Kir*(v_a-E_K_a)/(2+exp(1.62*(v_a-E_K_a)/V_T))*Hill(K_e,zeta_Kir,1)   : amp/meter**2
        I_NKP_a=I_NKA_a*Hill(N_a,zeta_Na,1.5)*Hill(K_e,zeta_K,1)/(1+0.1245*exp(-0.1*v_a/V_T)-0.0052*exp(-v_a/V_T)*(1-exp(N_e/67.3/mmolar))) : amp/meter**2
        I_NKCC=g_NKCC*(E_Na_a+E_K_a-2*E_Cl_a)             : amp/meter**2
    
        # Transporter currents
        E_EAAT=(3*E_Na_a+E_H_a-E_K_a-E_Glu_a)/2                 : volt
        E_GAT=(3*E_Na_a+E_Cl_a-V_T*log((GABA0_a/GABA_e)))/2     : volt
        # I_EAAT=g_EAAT*IversonBrackets((G_e-G0_e)/pmolar,1e-9)*(v_a-E_EAAT) : amp/meter**2
        I_EAAT=g_EAAT*int(G_e/G0_e>1.0)*(v_a-E_EAAT) : amp/meter**2
        # I_GAT=g_GAT*IversonBrackets((GABA_e-GABA0_e)/pmolar,1e-9)*(v_a-E_GAT)                           : amp/meter**2
        I_GAT=g_GAT*int(GABA_e/GABA0_e>1.0)*(v_a-E_GAT)                           : amp/meter**2
    
        # Leak current (ClCs + EAATs)
        I_Cl_a=(g_L_Cl_a + g_T_Cl_a*IversonBrackets(abs(I_EAAT/(amp/meter**2)),-1))*(v_a-E_Cl_a)    : amp/meter**2
    
        # GABA-mediated currents
        E_GABA_a=-V_T*log((C_e+P_HBC_Cl*HBC0_e)/(C_a+P_HBC_Cl*HBC0_a)) : volt
        I_GABA_a=g_GABA*r_GABA*(v_a-E_GABA_a)                 : amp/meter**2                
    
        # r.h.s. (astrocyte main (proximal) compartment)    
        dv_a/dt=(-pi_GABA_Cl*I_GABA_a-I_Cl_a-I_Kir-I_NKP_a-I_EAAT-2*I_GAT)/c_m_a : volt
        dn_Na_a/dt=-S_a*(3*I_NKP_a+3*I_EAAT-3*I_GAT-I_NKCC)/F + J_diff_Na*Lambda_a         : mole
        dn_K_a/dt=-S_a*(-I_Kir-2*I_NKP_a-I_EAAT-I_NKCC)/F + J_diff_K*Lambda_a              : mole
        dn_Cl_a/dt=S_a*(-I_GAT+2*I_NKCC+I_Cl_a+pi_GABA_Cl*I_GABA_a)/F + J_diff_Cl*Lambda_a : mole
    
        # Astrocyte-wide GABARs (assuming r_GABA=0 for ambient GABA0_e)              
        dr_GABA/dt = -r_GABA_clip/tau_GABA+J_GABA*(1-r_GABA_clip)*(GABA_e-GABA0_e) : 1        
        r_GABA_clip = clip(r_GABA,0,1)        : 1
        
        # Lumped Buffering (astrocyte)
        J_diff_Na = -D_Na_a*(N_a-N_a_d_clipped) : mmolar/second
        J_diff_K = -D_K_a*(K_a-K_a_d_clipped)   : mmolar/second
        J_diff_Cl = -D_Cl_a*(C_a-C_a_d_clipped) : mmolar/second
        
        # Distal Compartment 
        dN_a_d/dt = -J_diff_Na - J_m_Na : mmolar
        dK_a_d/dt = -J_diff_K  - J_m_K  : mmolar
        dC_a_d/dt = -J_diff_Cl - J_m_Cl : mmolar
        N_a_d_clipped = clip(N_a_d,0*mmolar,inf*mmolar)  : mmolar
        K_a_d_clipped = clip(K_a_d,0*mmolar,inf*mmolar)  : mmolar
        C_a_d_clipped = clip(C_a_d,0*mmolar,inf*mmolar)  : mmolar
        '''

    eqs_ecs = '''
            # ECS-related variations of moles (by diffusion from/to distal compartments)        
            N_e = clip(N0_e+(n_Na_e-n_Na_n-n_Na_a)/Lambda_e, 0*mmolar, inf*mmolar)  : mmolar
            K_e = clip(K0_e+(n_K_e-n_K_n-n_K_a)/Lambda_e, 0*mmolar, inf*mmolar)     : mmolar
            C_e = clip(C0_e+(n_Cl_e-n_Cl_n-n_Cl_a)/Lambda_e, 0*mmolar, inf*mmolar)  : mmolar

            # Lumped buffering (ECS)
            J_diff_Na_e = -D_Na_e*(N_e-N_e_d_clipped)  : mmolar/second
            J_diff_K_e = -D_K_e*(K_e-K_e_d_clipped)    : mmolar/second
            J_diff_Cl_e = -D_Cl_e*(C_e-C_e_d_clipped)  : mmolar/second

            # r.h.s.
            dn_Na_e/dt = J_diff_Na_e*Lambda_e        : mole
            dn_K_e/dt = J_diff_K_e*Lambda_e          : mole
            dn_Cl_e/dt = J_diff_Cl_e*Lambda_e        : mole

            # # Transmembrane fluxes at distal compartments
            J_m_Na = -D_Na_m*(N_e_d_clipped-N_a_d_clipped) : mmolar/second
            J_m_K = -D_K_m*(K_e_d_clipped-K_a_d_clipped)   : mmolar/second
            J_m_Cl = -D_Cl_m*(C_e_d_clipped-C_a_d_clipped) : mmolar/second

            # Distal Compartment (ECS)
            dN_e_d/dt = -J_diff_Na_e + J_m_Na : mmolar
            dK_e_d/dt = -J_diff_K_e + J_m_K   : mmolar
            dC_e_d/dt = -J_diff_Cl_e + J_m_Cl  : mmolar
            N_e_d_clipped = clip(N_e_d,0*mmolar,inf*mmolar)  : mmolar
            K_e_d_clipped = clip(K_e_d,0*mmolar,inf*mmolar)  : mmolar
            C_e_d_clipped = clip(C_e_d,0*mmolar,inf*mmolar)  : mmolar
            '''

    eqs_syn = '''
            # Relevant volumes
            Lambda_e : meter**3 (constant)
            Lambda_s : meter**3 (constant)
            
            # Resting Concentration
            G0_e    : mmolar (constant)
            GABA0_e : mmolar (constant)

            # ECS synaptic concentrations
            G_e = clip(G0_e+n_Glu_e/Lambda_e, 0*mmolar, inf*mmolar)        : mmolar
            GABA_e = clip(GABA0_e+n_GABA_e/Lambda_e, 0*mmolar, inf*mmolar) : mmolar

            # ECS-related Nt clearance by diffusion
            J_diff_Glu_e = -D_Glu_e*(G_e-G0_e)            : mmolar/second
            J_diff_GABA_e = -D_GABA_e*(GABA_e-GABA0_e)            : mmolar/second
            '''
    if sinput=='glu':
        eqs_syn += '''
            # Synaptic concentrations
            G_s = clip(G0_e+n_Glu_s/Lambda_s, 0*mmolar, inf*mmolar)        : mmolar

            # ECS-related variations in neurotransmitter concentrations from/to synapses
            J_diff_Glu_s = -D_Glu_e*(G_s-G_e)             : mmolar/second

            # ODEs        
            dn_Glu_s/dt = Lambda_s*J_diff_Glu_s - R_T*J_r*Lambda_s : mole
            dn_Glu_e/dt = S_a/F*I_EAAT - Lambda_s*J_diff_Glu_s + Lambda_e*J_diff_Glu_e: mole
            dn_GABA_e/dt = S_a/F*I_GAT + Lambda_e*J_diff_GABA_e : mole

            # Synaptic compartment
            J_r     : 1/second
            '''
    else:
        eqs_syn += '''
            # Synaptic concentrations
            GABA_s = clip(GABA0_e+n_GABA_s/Lambda_s, 0*mmolar, inf*mmolar) : mmolar

            # ECS-related variations in neurotransmitter concentrations from/to synapses
            J_diff_GABA_s = -D_GABA_e*(GABA_s-G_e)                : mmolar/second
            
            # ODEs
            dn_Glu_e/dt = Lambda_e*J_diff_Glu_e                      : mole        
            dn_GABA_s/dt = Lambda_s*J_diff_GABA_s - R_T*J_r*Lambda_s : mole
            dn_GABA_e/dt = S_a/F*I_GAT - Lambda_s*J_diff_GABA_s + Lambda_e*J_diff_GABA_e : mole
            
            # Synaptic compartment
            J_r     : 1/second
            '''

    ## Generate Equations
    eqs = Equations(eqs_neuron + eqs_ecs + eqs_syn + eqs_astro)

    ## Generate the neuron group
    nea = NeuronGroup(1, eqs,
                      # events=events,
                      namespace=params,
                      name=name,
                      method='euler',
                      dt=dt,
                      order=10)

    # Constants
    nea.GABA0_e = params['GABA0_e']
    nea.G0_e = params['G0_e']

    # Initialize variables
    nea.n_Na_n = 0*mole
    nea.n_K_n = 0*mole
    nea.n_Cl_n = 0*mole

    nea.n_Na_a = 0*mole
    nea.n_K_a = 0*mole
    nea.n_Cl_a = 0*mole
    nea.N_a_d = params['N0_a']
    nea.K_a_d = params['K0_a']
    nea.C_a_d = params['C0_a']
    nea.r_GABA = 0.

    nea.n_Na_e = 0*mole
    nea.n_K_e = 0*mole
    nea.n_Cl_e = 0*mole
    nea.N_e_d = params['N0_e']
    nea.K_e_d = params['K0_e']
    nea.C_e_d = params['C0_e']

    if sinput=='glu':
        nea.n_Glu_e = 0*mole
        nea.n_Glu_s = 0*mole
        nea.n_GABA_e = 0*mole
    else:
        nea.n_Glu_e = 0*mole
        nea.n_GABA_s = 0*mole
        nea.n_GABA_e = 0*mole

    nea.v = -70*mV
    nea.v_a = -90*mV

    return nea

# -----------------------------------------------------------------------------------------------------------------------
# Synaptic connections
# -----------------------------------------------------------------------------------------------------------------------
def synaptic_connection(stim_source, snc_target, params, sinput='glu', name='syn*', dt=0.1*us, delay=None):
    eqs = Equations('''
        # Bound fraction of postsynaptic receptors (assuming no activation at Nt_s=0)
        Nt_s : mmolar
        J_rec = -r_clip/tau_r+J*(1-r_clip)*Nt_s : 1/second
        J_r_post = J_rec : 1/second (summed)
        dr/dt = J_rec   : 1 (clock-driven)
        r_clip = clip(r,0,1)        : 1        
        ''')
    if sinput=='glu':
        eqs += Equations('''
                         G_AMPA_post = g*r_clip : siemens/meter**2 (summed)
                         ''')
        on_pre = '''
                 Nt_s = G_s_post              
                 n_Glu_s_post += Nt_rel*Lambda_s
                 '''
    else:
        eqs += Equations('''
                         G_GABA_post = g*r_clip : siemens/meter**2 (summed)
                         ''')
        on_pre = '''
                 Nt_s = GABA_s_post              
                 n_GABA_s_post += Nt_rel*Lambda_s
                 '''

    synapse = Synapses(stim_source, snc_target, eqs,
                       on_pre=on_pre,
                       namespace=params,
                       method='euler',
                       name=name,
                       delay=delay,
                       dt=dt,
                       order=0)

    return synapse

def nea_simulator(N_synapses,duration,
                  protocol='periodic',sinput='glu',
                  code_dir='./codegen/'):
    # Clean memory from previous builds (allowing multiple runs)
    device.delete(force=True)
    dt_sim = 0.05*us
    # Make sure that the number of incoming synaptic connections is an integer
    N_synapses = int(N_synapses)

    # Generate parameters
    pars_nea = nea_parameters()
    D = pars_nea['D_Glu_e'] if sinput=='glu' else pars_nea['D_GABA_e']
    pars_syn = synapse_parameters(D=D)
    pars_syn['Nt0'] = pars_nea['G0_e'] if sinput=='glu' else pars_nea['GABA0_e']
    pars_nea['R_T'] = pars_syn['R_T']

    # Generate synaptic connections
    if protocol=='periodic':
        stim = periodic_nodes(1,10*Hz,name='period*',dt=1*ms)
        i_pre = np.atleast_1d([0]).astype(int)
        j_pst = np.atleast_1d([0]).astype(int)
    elif protocol=='poisson':
        stim = poisson_nodes(N_synapses,10*Hz,trp=0.0*second,name='poiss*',dt=None)
        i_pre = np.arange(N_synapses).astype(int)
        j_pst = np.zeros(N_synapses).astype(int)
    assert sinput in ['glu', 'gaba'], "Transmitter type (ttype) can only be of 'glu' or 'gaba'"

    # Generate NEA node
    nea = nea_node(pars_nea,sinput=sinput,name='nea*',dt=dt_sim)
    nea.Lambda_e = pars_nea['Lambda_e']
    nea.Lambda_s = pars_syn['Lambda_s']

    # Set up synaptic connection
    syn = synaptic_connection(stim, nea, pars_syn, sinput=sinput, name='syn*', dt=dt_sim)
    syn.connect(i=i_pre, j=j_pst)
    syn.r = 0
    syn.Nt_s = 0*mmolar

    # Set up useful monitors
    mon_spk = SpikeMonitor(stim, record=True, name='spk')
    vnea = ['v','v_a','I_AMPA'] if sinput=='glu' else ['v','v_a','I_GABA']
    vnea += ['K_e','K_a','K_e_d','K_a_d']
    mon_nea = StateMonitor(nea,variables=vnea,record=True,dt=0.1*ms) if sinput=='glu' else StateMonitor(nea,variables=vnea,record=True,dt=0.1*ms)

    # Build Network
    network = Network([stim, nea, syn, mon_spk, mon_nea])

    ## Run the simulator
    network.run(duration=duration*second, report='text')
    device.build(directory=code_dir, clean=True)

    _, axs = plt.subplots(4, 1, figsize=(6, 10))
    axs[0].vlines(mon_spk.t_,mon_spk.i[:].T,mon_spk.i[:].T+0.9)
    axs[0].set(xticklabels='')
    axs[0].set_ylabel('Stimulation')

    # Synaptic current
    try:
        axs[1].plot(mon_nea.t_,mon_nea.I_AMPA[:].T/(nA/cm**2),'k-')
        axs[1].set(xticklabels='')
        axs[1].set_ylabel(r'$I_{AMPA}$ (nA/$\mu$m$^2$)')
    except:
        axs[1].plot(mon_nea.t_,mon_nea.I_GABA[:].T/(nA/cm**2),'k-')
        axs[1].set(xticklabels='')
        axs[1].set_ylabel(r'$I_{GABA}$ (nA/$\mu$m$^2$)')

    # Membrane Potentials
    axs[2].plot(mon_nea.t_,mon_nea.v[:].T/mV,'k-',label='N')
    axs[2].plot(mon_nea.t_,mon_nea.v_a[:].T/mV,'g-',label='A')
    axs[2].set_ylabel('v (mV)')
    axs[2].legend(loc='upper right')

    # Potassium Concentrations
    axs[3].plot(mon_nea.t_,mon_nea.K_e[:].T/mM,'b-',label='E')
    axs[3].plot(mon_nea.t_,mon_nea.K_e_d[:].T/mM,'b--',label=r'E$_\infty$')
    axs[3].plot(mon_nea.t_,mon_nea.K_a[:].T/mM,'c-',label='A')
    axs[3].plot(mon_nea.t_,mon_nea.K_a[:].T/mM,'c--',label=r'A$_\infty$')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_xlabel('K$^+$ (mM)')
    axs[3].legend(loc='upper right')

if __name__=="__main__":
    # -----------------------------------------------------------------------------------------------------------------------
    # Imports and Preamble
    # -----------------------------------------------------------------------------------------------------------------------
    import matplotlib.pyplot as plt

    code_dir = './codegen'
    prefs.GSL.directory = '/usr/include/'  ## The directory where the GSL library headings are found
    set_device('cpp_standalone',directory=code_dir,build_on_run=False)
    prefs.devices.cpp_standalone.openmp_threads = 2  ## The number of threads used in the parallelization (machine-dependent)
    prefs.logging.file_log = False
    prefs.logging.delete_log_on_exit = True

    # -----------------------------------------------------------------------------------------------------------------------
    # Testing
    # -----------------------------------------------------------------------------------------------------------------------
    nea_simulator(1,1.0,protocol='periodic',sinput='glu',code_dir='./codegen/')

    # -----------------------------------------------------------------------------------------------------------------------
    # Visualize
    # -----------------------------------------------------------------------------------------------------------------------
    plt.show()