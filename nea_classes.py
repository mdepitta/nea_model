import numpy as np
import scipy.constants as spc

# Custom modules
import os,sys

sys.path.append(os.path.join(os.path.expanduser('~'),'Ongoing.Projects/pycustommodules'))
import save_utils as svu
import general_utils as gu
import geometry as geom

# -----------------------------------------------------------------------------------------------------------------------
# Brian2 import: we use Brian CPP-standalone code generation for fast parallelized simulations
# -----------------------------------------------------------------------------------------------------------------------
from brian2 import *
from brian2.units.constants import faraday_constant as F
from functions_brianlib import ThermalPotential,NernstPotential,Hill,HeavisideFunction
import nea_parameters as neap

## Warning handling
import warnings as wrn
wrn.filterwarnings("ignore")
BrianLogger.suppress_name('resolution_conflict')

# -----------------------------------------------------------------------------------------------------------------------
# Dummy cell stimulation models
# -----------------------------------------------------------------------------------------------------------------------
def periodic_nodes(N,rates,name='p',dt=None):
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
def poisson_nodes(N,rates,trp=0.0*second,
        name='p',
        dt=None):
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
# Cell Models
# -----------------------------------------------------------------------------------------------------------------------
def neuron_cell(N,params,model='l5e',name='N*',method='rk4',dt=None):
    if model=='l5e':
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
    elif model=='l5i':
        pass

    # Add common section
    # TODO: Reduce redundancies when interneuron will be coded
    eqs += Equations('''
            # Parameters (explicit constants)
            N0_i     : mmolar (constant)
            K0_i     : mmolar (constant)
            C0_i     : mmolar (constant)
            HBC0_i   : mmolar (constant)
            HBC0_e   : mmolar (constant)
            Lambda   : meter**3 (constant)
            
            # ID
            neuron_id : 1 (constant)

            # Compute Intracellular Concentrations
            N_i=clip(N0_i+n_Na/Lambda,0*mmolar,inf*mmolar)  : mmolar
            K_i=clip(K0_i+n_K/Lambda,0*mmolar,inf*mmolar)   : mmolar
            C_i=clip(C0_i+n_Cl/Lambda,0*mmolar,inf*mmolar)  : mmolar

            # Extracellular concentrations
            N_e    : mmolar (linked)
            K_e    : mmolar (linked)
            C_e    : mmolar (linked)

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

            # Synaptic currents
            E_AMPA=V_T*log((N_e+P_K_Na*K_e)/(N_i+P_K_Na*K_i)) : volt
            E_GABA=-V_T*log((C_e+P_HBC_Cl*HBC0_e)/(C_i+P_HBC_Cl*HBC0_i)) : volt
            I_AMPA=G_AMPA*(v-E_AMPA)  : amp/meter**2
            I_GABA=G_GABA*(v-E_GABA)  : amp/meter**2
            G_AMPA : siemens/meter**2 (linked)
            G_GABA : siemens/meter**2 (linked)

            # ODEs
            dv/dt=(I_inj-I_AMPA-pi_GABA_Cl*I_GABA-I_Na-I_K-I_L-I_NKP)/c_m       : volt
            dn_Na/dt=-S*(-pi_AMPA_Na*I_AMPA+I_Na+I_L_Na+3*I_NKP)/F              : mole
            dn_K/dt=-S*(-pi_AMPA_K*I_AMPA+I_K+I_L_K-2*I_NKP+I_KCC)/F            : mole
            dn_Cl/dt=S*(pi_GABA_Cl*I_GABA+I_L_Cl+I_KCC)/F    : mole
        ''')

    # dv/dt = (I_inj-I_AMPA-I_Na-I_K-I_L-I_NKP)/c_m: volt
    # dn_Na_n/dt = -S_n*(-pi_AMPA_Na*I_AMPA+I_Na+I_L_Na+3*I_NKP)/F: mole
    # dn_K_n/dt = -S_n*(-pi_AMPA_K*I_AMPA+I_K+I_L_K-2*I_NKP+I_KCC)/F: mole
    # dn_Cl_n/dt = S_n*(I_L_Cl+I_KCC)/F: mole

    # Generate the neuron group
    neurons = NeuronGroup(N,eqs,
        threshold='v>v_thr',
        reset='',
        namespace=params,
        name=name,
        method=method,
        dt=dt)

    # Set constant parameters
    neurons.N0_i = params['N0_i']
    neurons.K0_i = params['K0_i']
    neurons.C0_i = params['C0_i']
    neurons.HBC0_i = params['HBC0_i']
    neurons.HBC0_e = params['HBC0_e']
    neurons.Lambda = params['Lambda']

    # Initialization of the variables
    neurons.m = 0.01
    neurons.h = 0.99
    neurons.n = 0.01
    neurons.n_Na = 0*mole
    neurons.n_K = 0*mole
    neurons.n_Cl = 0*mole

    return neurons

def astrocyte_cell(N,params,name='astro*',method='rk4',dt=None):
    eqs = Equations('''
        # Parameters (explicit constants)
        N0_a : mmolar (constant)
        K0_a : mmolar (constant)
        C0_a : mmolar (constant)
        HBC0_a  : mmolar (constant)
        HBC0_e  : mmolar (constant)
        H0_a    : mmolar (constant)
        H0_e    : mmolar (constant)
        G0_a    : mmolar (constant)
        G0_e    : mmolar (constant)
        GABA0_a : mmolar (constant)
        GABA0_e : mmolar (constant)
        Lambda  : meter**3 (constant)
        
        # Astrocyte ID
        astrocyte_id : 1 (constant)
        
        # Intracellular-related variations of moles (by diffusion from/to distal compartments)
        J_diff_Na    : mmolar/second
        J_diff_K     : mmolar/second
        J_diff_Cl    : mmolar/second
        
        # Concentrations
        N_a=clip(N0_a+n_Na/Lambda,0*mmolar,inf*mmolar)  : mmolar
        K_a=clip(K0_a+n_K/Lambda,0*mmolar,inf*mmolar)   : mmolar
        C_a=clip(C0_a+n_Cl/Lambda,0*mmolar,inf*mmolar)  : mmolar

        # External variables
        N_e    : mmolar (linked)
        K_e    : mmolar (linked)
        C_e    : mmolar (linked)
        G_e    : mmolar (linked)
        GABA_e : mmolar (linked)

        # Define relevant quantities
        V_T=ThermalPotential(T_exp)*volt                : volt
        E_Na=NernstPotential(N_e,N_a,1,T_exp)*volt      : volt 
        E_K=NernstPotential(K_e,K_a,1,T_exp)*volt       : volt 
        E_Cl=NernstPotential(C_e,C_a,-1,T_exp)*volt     : volt
        E_Glu=NernstPotential(G_e,G0_a,-1,T_exp)*volt   : volt
        E_H=NernstPotential(H0_e,H0_a,1,T_exp)*volt     : volt

        # Kir + XC currents
        I_Kir=g_Kir*(v-E_K)/(2+exp(1.62*(v-E_K)/V_T))*Hill(K_e,zeta_Kir,1)   : amp/meter**2
        I_NKP=I_NKA*Hill(N_a,zeta_Na,1.5)*Hill(K_e,zeta_K,1)/(1+0.1245*exp(-0.1*v/V_T)-0.0052*exp(-v/V_T)*(1-exp(N_e/67.3/mmolar))) : amp/meter**2
        I_NKCC=g_NKCC*(E_Na+E_K-2*E_Cl)                 : amp/meter**2
        
        # Transporter currents
        E_EAAT=(3*E_Na+E_H-E_K-E_Glu)/2                 : volt
        I_EAAT=g_EAAT*HeavisideFunction((G_e-G0_e)/nmolar,1e-6)*(v-E_EAAT)    : amp/meter**2
        E_GAT=(3*E_Na+E_Cl-V_T*log((GABA0_a/GABA_e)))/2 : volt
        I_GAT=g_GAT*HeavisideFunction((GABA_e-GABA0_e)/nmolar,1e-6)*(v-E_GAT) : amp/meter**2

        # Leak current (ClCs + EAATs)
        I_Cl=(g_L_Cl + g_T_Cl*HeavisideFunction(I_EEAT/(amp/meter**2),1e-9)*(v-E_Cl) : amp/meter**2

        # "Synaptic" currents
        E_GABA=-V_T*log((C_e+P_HBC_Cl*HBC0_e)/(C_a+P_HBC_Cl*HBC0_a)) : volt
        I_GABA=g_GABA*r_GABA*(v-E_GABA)   : amp/meter**2                
        # G_GABA                     : siemens/meter**2

        # r.h.s.    
        dv/dt=(-pi_GABA_Cl*I_GABA-I_Cl-I_Kir-I_NKP-I_EAAT-2*I_GAT)/c_m : volt
        dn_Na/dt=-S*(3*I_NKP+3*I_EAAT-3*I_GAT-I_NKCC)/F + J_diff_Na*Lambda         : mole
        dn_K/dt=-S*(-I_Kir-2*I_NKP-I_EAAT-I_NKCC)/F + J_diff_K*Lambda              : mole
        dn_Cl/dt=S*(-I_GAT+2*I_NKCC+I_Cl+pi_GABA_Cl*I_GABA)/F + J_diff_Cl*Lambda   : mole
        
        # Astrocyte-wide GABARs (w/ basal GABA activation)              
        dr_GABA/dt = -r_GABA_clipped/tau_GABA+w_GABA*(1-r_GABA_clipped)*(GABA_e-GABA0_e) : 1        
        r_GABA_clipped = clip(r_GABA,0,1)        : 1
    ''')

    astrocytes = NeuronGroup(N,eqs,
        namespace=params,
        name=name,
        method=method,
        dt=dt)

    # Set constant parameters
    astrocytes.N0_a = params['N0_a']
    astrocytes.K0_a = params['K0_a']
    astrocytes.C0_a = params['C0_a']
    astrocytes.HBC0_a = params['HBC0_a']
    astrocytes.HBC0_e = params['HBC0_e']
    astrocytes.H0_a = params['H0_a']
    astrocytes.H0_e = params['H0_e']
    astrocytes.G0_a = params['G0_a']
    astrocytes.G0_e = params['G0_e']
    astrocytes.GABA0_a = params['GABA0_a']
    astrocytes.GABA0_e = params['GABA0_e']
    astrocytes.Lambda = params['Lambda']
    # astrocytes.T_exp = params['T_exp']

    # Initial values
    astrocytes.n_Na = 0*mole
    astrocytes.n_K = 0*mole
    astrocytes.n_Cl = 0*mole

    return astrocytes

def ecs_space(N,params,coupling,name='ecs*',method='rk4',dt=None):

    eqs = '''
          # Parameters (explicit constants)
          N0_e     : mmolar (constant)
          K0_e     : mmolar (constant)
          C0_e     : mmolar (constant)
          HBC0_e   : mmolar (constant)
          G0_e     : mmolar (constant)
          GABA0_e  : mmolar (constant)
          Lambda_e : meter**3 (constant)

          # Adding linked variables 
          n_Na_i    : mole (linked)
          n_K_i     : mole (linked)
          n_Cl_i    : mole (linked)
          n_Na_a    : mole (linked)
          n_K_a     : mole (linked)
          n_Cl_a    : mole (linked)
          I_EAAT    : ampere/meter**2 (linked)
          I_GAT     : ampere/meter**2 (linked)   
          '''

    if coupling=='n':
        eqs = '\n'.join(eqs.splitlines()[:-6])
    elif coupling=='a':
        eqs = '\n'.join(eqs.splitlines()[:-9]) + '\n'.join(eqs.splitlines()[-7:])
    elif coupling=='none':
        eqs = '\n'.join(eqs.splitlines()[:-9])
    else:
        # this is the case of both, i.e. coupling=='na'
        pass

    eqs += '''
           # ECS-related diffusion fluxes from/to distal compartments
           J_diff_Na_e    : mmolar/second
           J_diff_K_e     : mmolar/second
           J_diff_Cl_e    : mmolar/second
           J_diff_Glu_e   : mmolar/second
           J_diff_GABA_e  : mmolar/second
           '''

    if coupling=='n':
        eqs += '''
               N_e = clip(N0_e+(n_Na_e-n_Na_i)/Lambda_e, 0*mmolar, inf*mmolar)  : mmolar
               K_e = clip(K0_e+(n_K_e-n_K_i)/Lambda_e, 0*mmolar, inf*mmolar)    : mmolar
               C_e = clip(C0_e+(n_Cl_e-n_Cl_i)/Lambda_e, 0*mmolar, inf*mmolar)  : mmolar
               '''
    elif coupling=='a':
        eqs += '''
               N_e = clip(N0_e+(n_Na_e-n_Na_a)/Lambda_e, 0*mmolar, inf*mmolar)  : mmolar
               K_e = clip(K0_e+(n_K_e-n_K_a)/Lambda_e, 0*mmolar, inf*mmolar)    : mmolar
               C_e = clip(C0_e+(n_Cl_e-n_Cl_a)/Lambda_e, 0*mmolar, inf*mmolar)  : mmolar
               '''
    elif coupling=='na':
        eqs += '''
               N_e = clip(N0_e+(n_Na_e-n_Na_i-n_Na_a)/Lambda_e, 0*mmolar, inf*mmolar)  : mmolar
               K_e = clip(K0_e+(n_K_e-n_K_i-n_K_a)/Lambda_e, 0*mmolar, inf*mmolar)     : mmolar
               C_e = clip(C0_e+(n_Cl_e-n_Cl_i-n_Cl_a)/Lambda_e, 0*mmolar, inf*mmolar)  : mmolar
               '''
    elif coupling=='none':
        eqs += '''
               N_e = clip(N0_e+n_Na_e/Lambda_e, 0*mmolar, inf*mmolar)  : mmolar
               K_e = clip(K0_e+n_K_e/Lambda_e, 0*mmolar, inf*mmolar)     : mmolar
               C_e = clip(C0_e+n_Cl_e/Lambda_e, 0*mmolar, inf*mmolar)  : mmolar
               '''
    eqs = Equations(eqs)
    eqs += Equations('''
         # Extracellular concentrations
         G_e = clip(G0_e+n_Glu_e/Lambda_e, 0*mmolar, inf*mmolar)        : mmolar
         GABA_e = clip(GABA0_e+n_GABA_e/Lambda_e, 0*mmolar, inf*mmolar) : mmolar

         # r.h.s.
         dn_Na_e/dt = J_diff_Na_e*Lambda_e      : mole
         dn_K_e/dt = J_diff_K_e*Lambda_e        : mole
         dn_Cl_e/dt = J_diff_Cl_e*Lambda_e      : mole
         
         # # Release events
         # # Glu_rel  : mmolar
         # # GABA_rel : mmolar
         # Glu_rel  : 1
         # GABA_rel : 1
         
         # ECS-related variations in neurotransmitter concentrations from/to synapses (or open-compartment leak when X_s is unassigned)
         J_diff_Glu_s  : mmolar/second
         J_diff_GABA_s : mmolar/second
 
         # ECS-related total neurotransmitter diffusion fluxes
         J_diff_Glu = J_diff_Glu_s + J_diff_Glu_e    : mmolar/second 
         J_diff_GABA = J_diff_GABA_s + J_diff_GABA_e : mmolar/second 
         
         # Synaptic Conductances
         G_AMPA : siemens/meter**2
         G_GABA : siemens/meter**2
        ''')

    if 'a' in coupling:
        eqs += Equations(''' 
            dn_Glu_e/dt = S_A/F*I_EAAT + Lambda_e*J_diff_Glu  : mole
            dn_GABA_e/dt = S_A/F*I_GAT + Lambda_e*J_diff_GABA : mole
            ''')
    else:
        eqs += Equations(''' 
            dn_Glu_e/dt = Lambda_e*J_diff_Glu                 : mole
            dn_GABA_e/dt = Lambda_e*J_diff_GABA               : mole
            ''')

    # # events = {'glu_release' : 'Glu_rel>0*mmolar',
    # #           'gaba_release': 'GABA_rel>0*mmolar'}
    # events = {'glu_release' : 'Glu_rel>0',
    #           'gaba_release': 'GABA_rel>0'}

    # Generate the neuron group
    ecs = NeuronGroup(N,eqs,
        # events=events,
        namespace=params,
        name=name,
        method=method,
        dt=dt)

    # Custom resets in relation to pre-synaptic releases
    # eqs_glu = '''
    #           n_Glu_e+= Glu_rel*Lambda_e
    #           Glu_rel=0*mmolar
    #           '''
    # eqs_GABA = '''
    #            n_GABA_e += GABA_rel*Lambda_e
    #            GABA_rel=0*mmolar
    #            '''
    # eqs_glu = '''Glu_rel=0'''
    # eqs_GABA = '''GABA_rel=0'''
    # ecs.run_on_event('glu_release',eqs_glu,when='before_groups',order=10)
    # ecs.run_on_event('gaba_release',eqs_GABA,when='before_groups',order=10)

    # Set constants
    ecs.N0_e = params['N0_e']
    ecs.K0_e = params['K0_e']
    ecs.C0_e = params['C0_e']
    ecs.HBC0_e = params['HBC0_e']
    ecs.G0_e = params['G0_e']
    ecs.GABA0_e = params['GABA0_e']
    ecs.Lambda_e = params['Lambda_e']

    # Initialization of variables
    ecs.n_Na_e = 0*mole
    ecs.n_K_e = 0*mole
    ecs.n_Cl_e = 0*mole
    ecs.n_Glu_e = 0*mole
    ecs.n_GABA_e = 0*mole
    # ecs.Glu_rel = 0*mmolar
    # ecs.GABA_rel = 0*mmolar
    # ecs.Glu_rel = 0
    # ecs.GABA_rel = 0

    return ecs

# -----------------------------------------------------------------------------------------------------------------------
# Connection models
# -----------------------------------------------------------------------------------------------------------------------
def synaptic_connection(stim_source, ecs_target, params, sinput='glu', name='syn*', dt=0.1*us, delay=None):
    eqs = Equations('''
        # Model Parameters
        g      : siemens/meter**2 (constant)
        Nt0_e  : mmolar (constant)
        Nt_rel : mmolar (constant)
        Lambda_s : meter**3 (constant)
        
        # Neurotransmitter concentrations 
        Nt_s = clip(Nt0_e+nt_s/Lambda_s, 0*mmolar, inf*mmolar) : mmolar

        # Fluxes
        J_rec = -r_clipped/tau_r+W*(1-r_clipped)*(Nt_s-Nt0_e)  : 1/second
        J_diff_syn = -D*(Nt_s-Nt_e)                            : mmolar/second
        
        # ODE
        dr/dt = J_rec                                 : 1 (clock-driven)
        dnt_s/dt = Lambda_s*(J_diff_syn - R_T*J_rec)  : mole (clock-driven)
        
        # Passing variables
        r_clipped = clip(r,0,1)        : 1                  
        ''')
    if sinput=='glu':
        eqs += Equations('''
                         J_diff_Glu_s_post = -J_diff_syn     : mmolar/second (summed)       
                         G_AMPA_post = g*r_clipped : siemens/meter**2 (summed)
                         Nt_e = G_e_post : mmolar
                         ''')
    else:
        eqs += Equations('''
                         J_diff_GABA_s_post = -J_diff_syn    : mmolar/second (summed)       
                         G_GABA_post = g*r_clipped : siemens/meter**2 (summed)
                         Nt_e = GABA_e_post : mmolar
                         ''')
    on_pre = '''
             nt_s += Nt_rel*Lambda_s
             '''

    synapse = Synapses(stim_source, ecs_target, eqs,
                       on_pre=on_pre,
                       namespace=params,
                       method='euler',
                       name=name,
                       delay=delay,
                       dt=dt,
                       order=0)

    return synapse

def ecs_ecs_connection(ecs_source,ecs_target,params,name='e2e*',dt=None):
    eqs = Equations('''
                    # rho = Lambda_e_pre/Lambda_e_post             : 1
                    J_diff_Na_e_pre = -D_Na_e*(N_e_pre-N_e_post)   : mmolar/second (summed)
                    J_diff_K_e_pre = -D_K_e*(K_e_pre-K_e_post)     : mmolar/second (summed)
                    J_diff_Cl_e_pre = -D_Cl_e*(C_e_pre-C_e_post)   : mmolar/second (summed)
                    J_diff_Glu_e_pre = -D_Glu_e*(G_e_pre-G_e_post) : mmolar/second (summed)
                    J_diff_GABA_e_pre = -D_GABA_e*(GABA_e_pre-GABA_e_post) : mmolar/second (summed)
                    J_diff_Na_e_post = D_Na_e*(N_e_pre-N_e_post)   : mmolar/second (summed)
                    J_diff_K_e_post = D_K_e*(K_e_pre-K_e_post)     : mmolar/second (summed)
                    J_diff_Cl_e_post = D_Cl_e*(C_e_pre-C_e_post)   : mmolar/second (summed)
                    J_diff_Glu_e_post = D_Glu_e*(G_e_pre-G_e_post) : mmolar/second (summed)
                    J_diff_GABA_e_post = D_GABA_e*(GABA_e_pre-GABA_e_post) : mmolar/second (summed)
                    ''')

    e2e = Synapses(ecs_source,ecs_target,eqs,
                   namespace=params,
                   name=name,dt=dt,
                   order=10)
    return e2e

def cell_ecs_connection(cell_source,ecs_target,source_id='neuron',name='c2e*',dt=None):
    """
    Wrapper to connect different cell compartments with corresponding ecs compartment

    - cell_source:
    - ecs_target:
    - source_id:

    Return : Does not return anything
    """

    """
    NOTE: The order of linking matters. e.g. if inside a N1 we have 
    
    y : linked
    x = f(y)
    
    and in N2 we have
    
    x : linked
    y = g(x)
     
    Start by linking x first and then y (reverse order w.r.t. N1 and N2).
    """

    if source_id!='na':
        # Neuron and Astrocyte variables
        if source_id in ['n','neuron']:
            # ECS variables
            ecs_target.n_Na_i = linked_var(cell_source.n_Na)
            ecs_target.n_K_i = linked_var(cell_source.n_K)
            ecs_target.n_Cl_i = linked_var(cell_source.n_Cl)
        elif source_id in ['a','astro','astrocyte']:
            # ECS variables
            ecs_target.n_Na_a = linked_var(cell_source.n_Na)
            ecs_target.n_K_a = linked_var(cell_source.n_K)
            ecs_target.n_Cl_a = linked_var(cell_source.n_Cl)
            ecs_target.I_EAAT = linked_var(cell_source.I_EAAT)
            ecs_target.I_GAT = linked_var(cell_source.I_GAT)

        # Cell variables
        cell_source.N_e = linked_var(ecs_target.N_e)
        cell_source.K_e = linked_var(ecs_target.K_e)
        cell_source.C_e = linked_var(ecs_target.C_e)
        if source_id in ['a','astro','astrocyte']:
            # Astrocyte variables
            cell_source.G_e = linked_var(ecs_target.G_e)
            cell_source.GABA_e = linked_var(ecs_target.GABA_e)
        elif source_id in ['n','neuron']:
            cell_source.G_AMPA = linked_var(ecs_target.G_AMPA)
            cell_source.G_GABA = linked_var(ecs_target.G_GABA)
    else:
        assert (len(cell_source)!=2), "cell_source=[neurons,astrocytes] for source_id=='na"
        neu_source = cell_source[0]
        ast_source = cell_source[1]
        # ECS variables
        ecs_target.n_Na_i = linked_var(neu_source.n_Na)
        ecs_target.n_K_i = linked_var(neu_source.n_K)
        ecs_target.n_Cl_i = linked_var(neu_source.n_Cl)
        ecs_target.n_Na_a = linked_var(ast_source.n_Na)
        ecs_target.n_K_a = linked_var(ast_source.n_K)
        ecs_target.n_Cl_a = linked_var(ast_source.n_Cl)
        ecs_target.I_EAAT = linked_var(ast_source.I_EAAT)
        ecs_target.I_GAT = linked_var(ast_source.I_GAT)

        # Neuron variables
        neu_source.N_e = linked_var(ecs_target.N_e)
        neu_source.K_e = linked_var(ecs_target.K_e)
        neu_source.C_e = linked_var(ecs_target.C_e)

        # Astro variables
        ast_source.N_e = linked_var(ecs_target.N_e)
        ast_source.K_e = linked_var(ecs_target.K_e)
        ast_source.C_e = linked_var(ecs_target.C_e)
        ast_source.G_e = linked_var(ecs_target.G_e)
        ast_source.GABA_e = linked_var(ecs_target.GABA_e)

    # if source_id in ['n','neuron','na']:
    #     eqs = Equations('''
    #          G_AMPA_pre = G_AMPA_s_post : siemens/meter**2 (summed)
    #          G_GABA_pre = G_GABA_s_post : siemens/meter**2 (summed)
    #      ''')
    #     if source_id in ['n','neuron']:
    #         neu_source = cell_source
    #     c2e = Synapses(neu_source,ecs_target,eqs,
    #                    method='euler',
    #                    name=name,
    #                    dt=dt,
    #                    order=5)
    #     return c2e
    # else:
    #     return None

def astro_astro_connection(astro_source,astro_target,params,name='a2a*',dt=None):
    eqs = Equations('''
            J_diff_Na_a_pre = -D_Na_a*(N_a_pre-N_a_post)   : mmolar/second (summed)
            J_diff_K_a_pre = -D_K_a*(K_a_pre-K_a_post)     : mmolar/second (summed)
            J_diff_Cl_a_pre = -D_Cl_a*(C_a_pre-C_a_post)   : mmolar/second (summed)
            J_diff_Na_a_post = D_Na_a*(N_a_pre-N_a_post)   : mmolar/second (summed)
            J_diff_K_a_post = D_K_a*(K_a_pre-K_a_post)     : mmolar/second (summed)
            J_diff_Cl_a_post = D_Cl_a*(C_a_pre-C_a_post)   : mmolar/second (summed)
    ''')

    a2a = Synapses(astro_source,astro_target,eqs,
                   namespace=params,
                   name=name,dt=dt,
                   order=10)

    return a2a

# -----------------------------------------------------------------------------------------------------------------------
# Test configurations
# -----------------------------------------------------------------------------------------------------------------------
def synaptic_connection_test(duration=0.5*second,code_dir='./codegen'):
    device.delete(force=True)

    # Generate parameters
    ttype = 'glu'
    # ttype = 'gaba'
    # pars_neuro = neap.neuron_parameters(I_NKA=0*nA/cm**2,g_KCC=0*uS/cm**2,I_inj=0*nA/cm**2,g_L_Na=0*uS/cm**2,g_L_K=0*uS/cm**2)
    pars_neuro = neap.neuron_parameters(I_inj=0*nA/cm**2,g_L_Na=0*uS/cm**2,g_L_K=0*uS/cm**2)
    pars_astro = neap.astrocyte_parameters(g_GABA=0*uS/cm**2,g_GAT=0*uS/cm**2,g_NKCC=0*uS/cm**2,g_L_Cl=0*uS/cm**2,g_Kir=0*uS/cm**2,
                                           g_T_L=0.2*uS/cm**2,g_EAAT=0.1*uS/cm**2,
                                           tau_GABA=5*ms,J_GABA=5e3/umolar/second,
                                           D_Na_a=0/second,D_K_a=0/second,D_Cl_a=0/second)
    pars_ecs = neap.ecs_parameters(D_Glu_e=100/second)
    pars_syn = neap.synapse_parameters(ttype=ttype,Nt_rel=1000*umolar,taud_Glu_e=5*ms,
                                       g=1*uS/cm**2,tau_r=10*ms,R_T=0*umolar)

    # # Update ecs parameters
    # tau = 10*ms
    # pars_ecs['D_Glu'] = 1/tau
    # pars_ecs['D_GABA'] = 1/tau

    # Update neuron parameters
    pars_neuro['HBC0_e'] = pars_ecs['HBC0_e']

    # Update astrocyte parameters
    pars_astro['HBC0_e'] = pars_ecs['HBC0_e']
    pars_astro['H0_e'] = pars_ecs['H0_e']
    pars_astro['G0_e'] = pars_ecs['G0_e']
    pars_astro['GABA0_e'] = pars_ecs['GABA0_e']

    # # Update ecs parameters
    # pars_ecs['S_A'] = pars_astro['S']
    # pars_ecs['T_exp'] = pars_astro['T_exp']
    # pars_ecs['g_EAAT'] = pars_astro['g_EAAT']
    # pars_ecs['g_GAT'] = pars_astro['g_GAT']

    # Update synaptic parameters
    pars_syn['D'] = pars_ecs['D_Glu_e'] if ttype=='glu' else pars_ecs['D_GABA_e']

    # Integrator parameters
    dt = 0.1*usecond

    # Generate neuronal compartments
    source = periodic_nodes(1,10*Hz,name='src*',dt=1*ms)
    ## Single neuron coupling
    source_id = 'n'
    # source_id = 'na'
    neuron = neuron_cell(1,pars_neuro,model='l5e',name='neu*',method='euler',dt=dt)
    neuron.v = -70*mV

    # Generate ecs compartments
    ecs = ecs_space(1,pars_ecs,coupling=source_id,name='ecs*',method='euler',dt=dt)

    # ## Single astrocyte coupling
    # source_id = 'a'
    # astro = astrocyte_cell(1,pars_astro,name='astro*',method='euler',dt=dt)
    # ecs = ecs_space(1,pars_ecs,coupling=source_id,name='ecs*',method='euler',dt=dt)
    #
    # # Generate a distal ecs compartment
    ecso = ecs_space(1,pars_ecs,coupling='none',name='ecso*',method='euler',dt=dt)

    #-------------------------------------------------------------------------------------------------------------------
    # Generate astrocyte and ecs compartments with distal configuration included
    #-------------------------------------------------------------------------------------------------------------------
    # astro_orphan = astrocyte_cell(1,pars_astro,name='astro*',method='euler',dt=dt)
    # # Connect
    # astro_orphan.v = -60*mV
    # # astro.G0_a = pars_astro['G0_a']
    # astro_orphan.GABA0_e = pars_ecs['GABA0_e']
    # # Connect with the previous astrocyte
    # a2a = astro_astro_connection(astro,astro_orphan)
    # source_id = 'a'
    # astro = astrocyte_cell(2,pars_astro,name='astro*',method='euler',dt=dt)
    # ecs = ecs_space(2,pars_ecs,coupling=source_id,name='ecs*',method='euler',dt=dt)

    #-------------------------------------------------------------------------------------------------------------------
    # # Connect neuron with ecs
    #-------------------------------------------------------------------------------------------------------------------
    # c2e = cell_ecs_connection(neuron,ecs,source_id=source_id,name='c2e*',dt=dt)
    # c2e.connect(i=[0],j=[0])
    cell_ecs_connection(neuron,ecs,source_id=source_id,name='c2e*',dt=dt)

    #-------------------------------------------------------------------------------------------------------------------
    # Connect ECS compartments
    #-------------------------------------------------------------------------------------------------------------------
    e2e = ecs_ecs_connection(ecs,ecso,pars_ecs,name='e2e*',dt=dt)
    e2e.connect(i=[0],j=[0])

    # Connect neuron with ecs
    # c2e = cell_ecs_connection(astro,ecs,source_id=source_id,name='c2e*',dt=dt)
    # n2e = cell_ecs_connection([neuron,astro],ecs,source_id='na',name='n2e*',dt=dt)
    # a2e = cell_ecs_connection(astro,ecs,source_id='a',name='a2e*',dt=dt)
    # Neuron and Astrocyte variables
    # # ECS variables
    # ecs.n_Na_i = linked_var(neuron.n_Na)
    # ecs.n_K_i = linked_var(neuron.n_K)
    # ecs.n_Cl_i = linked_var(neuron.n_Cl)
    # # ECS variables
    # ecs.n_Na_a = linked_var(astro.n_Na)
    # ecs.n_K_a = linked_var(astro.n_K)
    # ecs.n_Cl_a = linked_var(astro.n_Cl)
    # ecs.I_EAAT = linked_var(astro.I_EAAT)
    # ecs.I_GAT = linked_var(astro.I_GAT)
    #
    # # Cell variables
    # neuron.N_e = linked_var(ecs.N_e)
    # neuron.K_e = linked_var(ecs.K_e)
    # neuron.C_e = linked_var(ecs.C_e)
    # astro.N_e = linked_var(ecs.N_e)
    # astro.K_e = linked_var(ecs.K_e)
    # astro.C_e = linked_var(ecs.C_e)
    #
    # # Astrocyte variables
    # astro.G_e = linked_var(ecs.G_e)
    # astro.GABA_e = linked_var(ecs.GABA_e)


    # # Connect ecs with ecs (always after linkages
    # e2e = ecs_ecs_connection(ecs,ecs_orphan,pars_ecs,name='e2e*',dt=dt)
    # e2e.connect(i=[0],j=[0])
    # Distal compartments
    # e2e = ecs_ecs_connection(ecs,ecs,pars_ecs,name='e2e*',dt=dt)
    # e2e.connect(i=[0,1],j=[1,0])
    # a2a = astro_astro_connection(astro,astro,pars_astro,name='a2a*',dt=dt)
    # a2a.connect(i=[0,1],j=[1,0])

    #-------------------------------------------------------------------------------------------------------------------
    # Synaptic connection
    #-------------------------------------------------------------------------------------------------------------------
    syn = synaptic_connection(source,ecs,pars_syn,sinput=ttype,name='syn*',dt=dt,delay=None)
    syn.connect(i=[0],j=[0])

    # Assign default parameters
    syn.g = pars_syn['g'] ## NOTE: This g also reflect the g in g_GABA for the astrocyte!!!
    syn.Nt_rel = pars_syn['Nt_rel']
    syn.Lambda_s = pars_syn['Lambda_s']
    syn.Nt0_e = pars_ecs['G0_e'] if ttype=='glu' else pars_ecs['GABA0_e']

    # Set initial conditions
    syn.r = 0.0
    syn.nt_s = 0.0*mole

    # # Initialize Neurons
    # neuron.v = NernstPotential(pars_ecs['C0_e'],pars_neuro['C0_i'],-1,pars_neuro['T_exp'])*volt
    # # Initialize Astrocyte
    # astro.v = NernstPotential(pars_ecs['K0_e'],pars_astro['K0_a'],1,pars_astro['T_exp'])*volt
    # astro.v = -60*mV
    # astro.GABA0_e = pars_ecs['GABA0_e']
    # astro.G0_e = pars_ecs['G0_e']

    #-----------------------------------------------------------------------------------------------------------------------
    # Add Monitors
    #-----------------------------------------------------------------------------------------------------------------------
    # Spiking Monitor
    #-----------------------------------------------------------------------------------------------------------------------
    spk_mon = SpikeMonitor(source, record=True)

    # -----------------------------------------------------------------------------------------------------------------------
    # Neuron Monitors
    #-----------------------------------------------------------------------------------------------------------------------
    neu_mon = StateMonitor(neuron,variables=['v','N_i','K_i','C_i'],record=True,dt=0.5*msecond)

    #-----------------------------------------------------------------------------------------------------------------------
    # ECS Monitors
    #-----------------------------------------------------------------------------------------------------------------------
    ecs_mon = StateMonitor(ecs,variables=['G_e','N_e','J_diff_Glu_e'],record=True,dt=0.5*msecond)
    ecso_mon = StateMonitor(ecso,variables=['G_e'],record=True,dt=0.5*msecond)


    #-----------------------------------------------------------------------------------------------------------------------
    # Synapse Monitors
    #-----------------------------------------------------------------------------------------------------------------------
    syn_mon = StateMonitor(syn,variables=['Nt_s','r'],record=True,dt=0.5*msecond)

    # #-----------------------------------------------------------------------------------------------------------------------
    # # Add Monitors (related to neuron)
    # # spk_mon = SpikeMonitor(source,record=True)
    # neu_mon = StateMonitor(neuron,variables=['v','I_GABA','E_GABA','N_i','K_i','C_i'],record=True,dt=0.5*msecond)
    # ecs_mon = StateMonitor(ecs,variables=['G_e'],record=True,dt=0.5*msecond,when='resets')
    # # ecso_mon = StateMonitor(ecs_orphan,variables=['G_e'],record=True,dt=0.5*msecond,when='resets')
    # # glu_mon = EventMonitor(ecs,event='glu_release',variables=['Glu_rel'],when='start',order=0)
    # # gaba_mon = EventMonitor(ecs,event='gaba_release',variables=['GABA_rel'],when='start',order=0)
    # syn_mon = StateMonitor(syn,variables=['r','nt_s'],record=True,dt=0.5*msecond)
    #
    # # Generate network
    # ## GLU/ECSO
    # network = Network([source,neuron,ecs,ecs_orphan,syn,c2e,e2e,neu_mon,ecs_mon,syn_mon])
    # ## GABA
    # ## network = Network([source,neuron,ecs,ecs_orphan,c2e,e2e,syn,neu_mon,gaba_mon,ecs_mon])
    # ## Standard
    # # network = Network([source,neuron,ecs,ecs_orphan,c2e,e2e,syn,neu_mon])
    # # network = Network([source,neuron,ecs])
    # # -----------------------------------------------------------------------------------------------------------------------
    # Astrocyte Monitors
    # ast_mon = StateMonitor(astro,variables=['v','E_EAAT','I_Cl','N_a','K_a','C_a'],record=True,dt=0.5*msecond)


    # ecs_mon = StateMonitor(ecs,variables=['G_e','G_s'],record=True,dt=0.5*msecond,when='resets')
    # network = Network([source,astro,ecs,ecs_orphan,c2e,e2e,syn,ast_mon,ecs_mon])
    # network = Network([source,astro,ecs,ecs_orphan,e2e,syn,ast_mon,ecs_mon])
    # network = Network([source,astro,ecs,e2e,a2a,syn,ast_mon,ecs_mon])
    # # -----------------------------------------------------------------------------------------------------------------------
    # Both neuron and astrocyte
    # network = Network([source,neuron,astro,ecs,n2e,e2e,a2a,syn,ast_mon,ecs_mon])
    # network = Network([source, neuron, astro, ecs, n2e, e2e, syn, ast_mon, ecs_mon])
    # network = Network([source, neuron, astro, ecs, n2e, syn, ast_mon, ecs_mon])
    # network = Network([source,neuron,astro,ecs,e2e,a2a,syn,ast_mon,ecs_mon])

    network = Network([source,neuron,ecs,ecso,e2e,syn,neu_mon,ecs_mon,ecso_mon,syn_mon])

    ## Run the simulator
    network.run(duration=duration*second,report='text')
    device.build(directory=code_dir,clean=True)

    # # # Plotting (Neuron)
    fig,axs = plt.subplots(4,1)
    axs[0].plot(neu_mon.t_,neu_mon.v[:].T/mV,'k-')
    # # axs[0].plot(neu_mon.t_,neu_mon.E_AMPA[:].T,'y-')
    # axs[0].plot(neu_mon.t_,neu_mon.E_GABA[:].T,'y-')
    # # # axs[0].vlines(spk_mon.t_,spk_mon.i[:].T,spk_mon.i[:].T+0.9)
    # # # axs[1].vlines(glu_mon.t_,glu_mon.Glu_rel[:].T,'ko')
    # # axs[1].plot(neu_mon.t_,neu_mon.I_AMPA[:].T,'r-')
    # axs[1].plot(neu_mon.t_,neu_mon.I_GABA[:].T,'r-')
    # # # axs[1].plot(ecs_mon.t_,ecs_mon.GABA_e[:].T,'r-')
    # # axs[1].plot(ecs_mon.t_,ecs_mon.G_e[:].T,'k-')
    axs[1].plot(neu_mon.t_,neu_mon.N_i[:].T/mmolar,'b-')
    # # # axs[2].plot(ecs_mon.t_,ecs_mon.G_e[:].T,'r-')
    # # # axs[2].plot(ecso_mon.t_,ecso_mon.G_e[:].T,'b-')
    # # # axs[2].plot(ecs_diff_mon.t_,ecs_diff_mon.Delta_n_Glu_tot[:].T,'b-')
    # # # axs[2].plot(gaba_mon.t_,gaba_mon.GABA_rel[:].T,'bo')
    # axs[2].plot(neu_mon.t_,neu_mon.K_i[:].T,'g-')
    #-----------------------
    # axs[2].plot(syn_mon.t_,syn_mon.Nt_s[:].T/umolar,'g-')
    axs[2].plot(ecs_mon.t_, ecs_mon.N_e[:].T/mmolar, 'g-')
    # axs[3].plot(syn_mon.t_,syn_mon.r[:].T,'g-')
    # -----------------------
    # axs[3].plot(ecs_mon.t_,ecs_mon.G_e[:].T/mmolar,'m-')
    # axs[3].plot(ecs_mon.t_, ecs_mon.J_diff_Glu_s[:].T/(nmolar/second), 'm-')
    # axs[3].plot(ecs_mon.t_, ecs_mon.J_diff_Glu_e[:].T/(nmolar/second), 'm-')
    axs[3].plot(ecso_mon.t_, ecs_mon.G_e[:].T/umolar, 'm-')
    # # # print(glu_mon.Glu_rel[:].T)

    # fig,axs = plt.subplots(4,1)
    # axs[0].plot(ast_mon.t_,ast_mon.v[:].T,'k-')
    # # axs[0].plot(ast_mon.t_,ast_mon.E_K[:].T,'y-')
    # # axs[0].plot(ast_mon.t_,ast_mon.E_GAT[:].T,'y-')
    # # axs[1].plot(ast_mon.t_,ast_mon.I_GABA[:].T,'r-')
    # # axs[0].plot(ast_mon.t_,ast_mon.E_GABA[:].T,'y-')
    # axs[0].plot(ast_mon.t_, ast_mon.E_EAAT[:].T, 'y-')
    # # axs[1].plot(ast_mon.t_, ast_mon.I_GAT[:].T, 'r-')
    # # axs[2].plot(ecs_mon.t_,ecs_mon.GABA_e[:].T,'b-')
    # # axs[1].plot(ast_mon.t_,ast_mon.N_a[:].T,'g-')
    # # axs[1].plot(ecs_mon.t_,ecs_mon.GABA_e[:].T,'g-')
    # axs[1].plot(ecs_mon.t_, ecs_mon.G_e[0,:].T, 'g-')
    # axs[1].plot(ecs_mon.t_, ecs_mon.G_s[0,:].T, 'b-')
    # # axs[2].plot(ast_mon.t_,ast_mon.K_a[:].T,'b-')
    # axs[2].plot(ast_mon.t_,ast_mon.K_a[:].T,'r-')
    # axs[3].plot(ast_mon.t_,ast_mon.I_Cl[:].T,'m-')

def simple_testing_code():
    device.delete(force=True)
    eqs_1 = Equations('''
            y : 1 (linked)
            dx/dt = (-x + a*x*y)/tau_x : 1
            ''')
    eqs_2 = Equations('''
            x : 1 (linked)
            dy/dt = (2*cos(x)-y_clip)/tau_y  : 1
            y_clip = clip(y0+y+x,-1,1) : 1
            ''')

    pars_1 = {'a' : 0.5, 'tau_x': 0.1*second}
    pars_2 = {'y0': 0.1, 'tau_y': 0.5*second}

    N1 = NeuronGroup(1,eqs_1,method='euler',dt=0.1*ms,name='N1',namespace=pars_1)
    N2 = NeuronGroup(1,eqs_2,method='euler',dt=0.1*ms,name='N2',namespace=pars_2)
    N1.y = linked_var(N2.y)
    N2.x = linked_var(N1.x)
    network = Network([N1,N2])
    network.run(duration=1*second,report='text')
    device.build(directory=code_dir,clean=True)

def simple_test():
    device.delete(force=True)

    # Generate parameters
    pars_neuro = neap.neuron_parameters()
    pars_ecs = neap.ecs_parameters()
    pars_syn = neap.synapse_parameters()

    # Update parameters
    pars_neuro['HBC0_e'] = pars_ecs['HBC0_e']

    # Integrator parameters
    dt = 0.01*msecond

    # Generate Compartments
    source = periodic_nodes(1,10*Hz,name='src',dt=1*ms)

    eqs = Equations('''
                # Compute Intracellular Concentrations
                N_i=clip(N0_i+n_Na/Lambda,0*mmolar,inf*mmolar)  : mmolar
                K_i=clip(K0_i+n_K/Lambda,0*mmolar,inf*mmolar)   : mmolar
                C_i=clip(C0_i+n_Cl/Lambda,0*mmolar,inf*mmolar)  : mmolar

                # Extracellular concentrations
                N_e    : mmolar (linked)
                K_e    : mmolar (linked)
                C_e    : mmolar (linked)
                
                dn_Na/dt=0*mole/second              : mole
                dn_K/dt=0*mole/second               : mole
                dn_Cl/dt=0*mole/second              : mole
                ''')
    neuron = NeuronGroup(1,eqs,name='neu',method='euler',dt=0.1*ms,namespace=pars_neuro)

    eqs = Equations('''
             # Cell parameters
             Lambda_e  : meter**3 (constant)

             # Cell-related variations of moles (by transmembrane fluxes)
             n_Na_i    : mole (linked)
             n_K_i     : mole (linked)
             n_Cl_i    : mole (linked)

             # Extracellular concentrations
             N_e = clip(N0_e+(n_Na_e-n_Na_i)/Lambda_e, 0*mmolar, inf*mmolar)  : mmolar
             K_e = clip(K0_e+(n_K_e-n_K_i)/Lambda_e, 0*mmolar, inf*mmolar)     : mmolar
             C_e = clip(C0_e+(n_Cl_e-n_Cl_i)/Lambda_e, 0*mmolar, inf*mmolar)  : mmolar

             # r.h.s.
             dn_Na_e/dt = 0*mole/second      : mole
             dn_K_e/dt = 0*mole/second       : mole
             dn_Cl_e/dt = 0*mole/second      : mole
             ''')
    ecs = NeuronGroup(1,eqs,name='ecs',dt=0.1*ms,method='euler',namespace=pars_ecs)

    ecs.n_Na_i = linked_var(neuron.n_Na)
    ecs.n_K_i = linked_var(neuron.n_K)
    ecs.n_Cl_i = linked_var(neuron.n_Cl)
    neuron.N_e = linked_var(ecs.N_e)
    neuron.K_e = linked_var(ecs.K_e)
    neuron.C_e = linked_var(ecs.C_e)

    network = Network([source,neuron,ecs])
    network.run(duration=1*second,report='text')
    device.build(directory=code_dir,clean=True)
# -----------------------------------------------------------------------------------------------------------------------
# Test configurations
# -----------------------------------------------------------------------------------------------------------------------
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
    synaptic_connection_test(duration=0.3,code_dir='./codegen')
    # dummy1()
    # simple_testing_code()
    # simple_test()   ## This one compiles ok -- but the code seems exactly the same

    # -----------------------------------------------------------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------------------------------------------------------
    plt.show()