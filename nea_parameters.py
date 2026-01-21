import numpy as np
import scipy.constants as spc
import copy as cp

# Custom modules
import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'),'Ongoing.Projects/pycustommodules'))
import general_utils as gu
import ngn_topology as conn

# Brian2 imports
# from brian2.units.allunits import *
from brian2.units import *
from brian2.units.constants import avogadro_constant as N_A
from functions_brianlib import NernstPotential,ThermalPotential

#-----------------------------------------------------------------------------------------------------------------------
# Parameter generator
#-----------------------------------------------------------------------------------------------------------------------
def ecs_parameters(**kwargs):
    pars = {# Extracellular concentrations
            'N0_e'   : 145*mmolar,
            'K0_e'   : 3*mmolar,
            'C0_e'   : 130*mmolar,
            'HBC0_e'  : 35*mmolar,
            'H0_e'    : 50*nmolar,
            'G0_e'    : 25*nmolar,
            'GABA0_e' : 50*nmolar,
            # Diffusion Rates
            'taud_Na_e'   : 1*msecond,
            'taud_K_e'    : 1*msecond,
            'taud_Cl_e'   : 1*msecond,
            'taud_Glu_e'  : 20*msecond,
            'taud_GABA_e' : 20*msecond,
            # Geometry
            'Lambda_e' : 500*um**3,
    }

    # Define diffusion rates
    for k in ['Na','K','Cl','Glu','GABA']:
        pars['D_'+k+'_e'] = 1.0/pars['taud_'+k+'_e']

    pars = gu.varargin(pars,**kwargs)

    return pars

def synapse_parameters(ttype='glu',**kwargs):
    assert any(ttype==t for t in ['glu','gaba']),"Allowed transmitter types (ttype): 'glu' or 'gaba' only"
    if ttype=='glu':
        pars = {'Nt_rel' : 0.1*mmolar,
                'W'      : 1/umolar/second,
                'tau_r'  : 10*msecond,
                'g'      : 10*nsiemens/cm**2,
                'taud_Glu_e': 20*msecond,
        }
    elif ttype=='gaba':
        pars = {'Nt_rel' : 1.0*mmolar,
                'W'      : 1/umolar/second,
                'tau_r'  : 50*msecond,
                'g'      : 10*nsiemens/cm**2,
                'taud_GABA_e': 20*msecond,
                }

    pars['D'] = 1.0/pars['taud_Glu_e'] if ttype=='glu' else 1.0/pars['taud_GABA_e']
    pars['Lambda_s'] = 8.75e-3*um**3
    # The current S_s and sigma_R are temporarily taken from --> need to be better estimated
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130924
    pars['S_s'] = 200*nmeter
    pars['sigma_R'] = 1000*um**-2
    # Estimate receptor density
    pars['R_T'] = pars['sigma_R']/N_A/pars['Lambda_s']*pars['S_s']**2


    pars = gu.varargin(pars,**kwargs)
    return pars

def neuron_parameters(model='l5e',**kwargs):
    pars = {## Concentrations to setup reverse potentials
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
            'S'      : 1750*um**2,
            'Lambda' : 525*um**3,
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

    pars = gu.varargin(pars,**kwargs)

    # Retrieve partial permeabilities
    pars['pi_AMPA_Na'] = 1.0/(1+pars['P_K_Na'])
    pars['pi_AMPA_K'] = pars['P_K_Na']/(1+pars['P_K_Na'])
    pars['pi_GABA_Cl'] = 1.0/(1+pars['P_HBC_Cl'])

    # # Extrapolate useful variables
    pars['T_adj'] = 2.3**(0.1*(pars['T_exp']-21))

    return pars

def astrocyte_parameters(**kwargs):
    pars = {## Concentrations to setup reverse potentials
            'N0_a'    : 15*mmolar,
            'K0_a'    : 100*mmolar,
            'C0_a'    : 40*mmolar,
            'H0_a'    : 60*nmolar,
            'G0_a'    : 25*nmolar, ## Glutamate
            'GABA0_a' : 20*mmolar, ## GABA ()
            'HBC0_a'  : 10*mmolar,
            ## Astrocyte Parameters
            'c_m'     : 1*ufarad/cm**2,
            ## Temperature
            'T_exp'   : 37, ## Body temperature of the animal
            ## External Stimulation
            'I_dc'    : 0*namp/cm**2,
            'I_ramp'  : 0*namp/cm**2,
            'T_ramp'  : 100*second,
            ## Geometry
            'S'       : 850*um**2,
            'Lambda'  : 1000*um**3,
            ## Kir
            'g_Kir'   : 175*usiemens/cm**2,
            'zeta_Kir': 13*mmolar,
            ## EAAT
            'sigma_EAAT': 100/um**2,
            'g_EAAT'    : 0*usiemens/cm**2,
            'g_T_Cl'    : 0*usiemens/cm**2,
            'g_L_Cl'    : 0*usiemens/cm**2,
            'Omega_Glu' : 25/second,
            ## GAT
            'g_GAT'   : 0*usiemens/cm**2,
            ## NKCC
            'g_NKCC'  : 0*usiemens/cm**2,
            ## NKP
            'I_NKA'   : 0*namp/cm**2,
            'zeta_Na' : 10*mmolar,
            'zeta_K'  : 3*mmolar,
            ## GABA
            'g_GABA'  : 0*usiemens/cm**2,
            'P_HBC_Cl': 0.4,
            'tau_GABA': 50*ms,
            'J_GABA'  : 1/umolar/second,
            ## Intracellular diffusion
            'taud_Na_a': 1*msecond,
            'taud_K_a': 1*msecond,
            'taud_Cl_a': 1*msecond,
    }

    # Define diffusion rates
    for k in ['Na','K','Cl']:
        pars['D_'+k+'_a'] = 1.0/pars['taud_'+k+'_a']

    pars = gu.varargin(pars,**kwargs)

    # Compute partial permeabilities
    pars['pi_GABA_Cl'] = 1.0/(1+pars['P_HBC_Cl'])

    # Extrapolate useful variables
    pars['T_adj'] = 2.3**(0.1*(pars['T_exp']-21))

    return pars


def geometry_parameters(N_e,N_i,N_g,geometry,**kwargs):
    """
    Build dictionary of spatial_geometry parameters used by ngn_connections() method

    Input arguments:
    - geometry  : {None} | 'planar' | 'spherical'

    Return:
    - spg  : dictionary with entries for geometrical/physical parameters
    """

    pgeom = {'id': None,
             # The following are parameters used in the spatial network configurations
             'syn_surface_density': 1.,  # in synapse/um**2 (from ...)
             'syn_volume_density': 1/1.13,  # in synapse/um**3 (from Katshuri et al., Cell 2015)
             'glia_distance': 80.,  # in um (from Chao et al., Chapter 1, Tripartite Synapse)
             # The following parameters are used in the spatial configuration to retrieve delays
             'conduction_speed': 0.25,  # in m/s (for neurons) (from Gerstner's book, p. 73)
             'calcium_speed': 25.0,  # in um/s (for glial) (from Poskanzer NN2019)
             # Cluster handling
             'N_clusters': 0,  # Number of clusters (Default: no clusters)
             'e_cluster' : [], # List of clusters in terms of neurons E
             'i_cluster' : [], # List of clusters in terms of neurons I
             'g_cluster' : [], # List of clusters in terms of neurons G
             # Number of cells per cluster(s)
             'N_e_cluster': 0,
             'N_i_cluster': 0,
             'N_g_cluster': 0,
            }

    # Custom parameters
    pgeom = gu.varargin(pgeom,**gu.merge_dicts({'id': geometry},kwargs))

    if pgeom['id'] == 'planar':
        pgeom.pop('syn_volume_density')
        pgeom['syn_density'] = pgeom.pop('syn_surface_density')
    elif pgeom['id'] == 'spherical':
        pgeom.pop('syn_surface_density')
        pgeom['syn_density'] = pgeom.pop('syn_volume_density')

    # Handling of pre-defined clustering (only if N_clusters>=1)
    if pgeom['N_clusters']>0:
        pgeom['N_clusters'] = int(pgeom['N_clusters'])
        assert (pgeom['N_clusters']<=N_e),"N_clusters must be <= N_e"
        assert (pgeom['N_clusters']<=N_i), "N_clusters must be <= N_i"
        # If there is only one cluster, the vector of the indices of the neurons in the cluster is just the arange(N_neuron)
        pgeom['e_cluster'] = np.arange(N_e)
        pgeom['i_cluster'] = np.arange(N_i)
        if N_g>0:
            assert (pgeom['N_clusters']<=N_g), "N_clusters must be <= N_g"
            pgeom['g_cluster'] = np.arange(N_g)
        if pgeom['N_clusters']>1:
            # # Generate random clusters (DISABLED till when Brian will allow recordings of non-contiguous subgroups)
            # np.random.shuffle(pgeom['e_cluster'])
            # np.random.shuffle(pgeom['i_cluster'])
            pgeom['e_cluster'] = np.array_split(pgeom['e_cluster'],pgeom['N_clusters'])
            pgeom['i_cluster'] = np.array_split(pgeom['i_cluster'],pgeom['N_clusters'])
            if N_g>0:
                # np.random.shuffle(pgeom['g_cluster'])
                pgeom['g_cluster'] = np.array_split(pgeom['g_cluster'],pgeom['N_clusters'])

        for cell in ['e','i','g']:
            # Ensure 2D arrays for proper handling (also for the exceptional case of N_cluster==1)
            pgeom[cell + '_cluster'] = np.atleast_2d(pgeom[cell + '_cluster'])
            # Update cluster sizes
            pgeom['N_'+cell+'_cluster'] = np.asarray([np.size(c) for c in pgeom[cell+'_cluster']]).astype(np.int32)

    return pgeom

# -----------------------------------------------------------------------------------------------------------------------
# Geometry Parameters (currently copied from ngn_parameters)
# -----------------------------------------------------------------------------------------------------------------------
def set_module_parameters(module_pars='default', module_default={}, module_same={}):
    """
    Set parameters of a specific module.

    WARNING: Depending on value of module_pars, module_default is a different object!! (See Input argument specification for
    further details).

    Args:
    - module_pars    : {'default'} | 'same' | Dictionary
    - module_default : Default module parameter dictionary
    - module_same    : Default module parameter dictionary for 'same'

    Returns:
    - pars           : Dictionary of parameters

    Maurizio De Pitta', The University of Chicago, May 27th 2016.
    """

    # NOTE: Deepcopy assures that you are not passing by reference
    if module_pars == 'default':
        assert type(
            module_default) == dict, "'default' module_pars requires passing module specific parameter dictionary"
        pars = cp.deepcopy(module_default)
    elif module_pars == 'same':
        assert type(module_same) == dict, "'same' module_pars requires passing module specific parameter dictionary"
        pars = cp.deepcopy(module_same)
    elif type(module_pars) == dict:
        # assert hasattr(pars_gen, '__call__'), "module_pars dict must be passed to module specific parameter generating function"
        assert type(
            module_default) == dict, "'default' module_pars requires passing module specific parameter dictionary"
        pars = gu.merge_dicts(cp.deepcopy(module_default), module_pars)
    return pars

def set_weights(w_val, w_rule=None, N=1):
    """
    Set weight w_val according to w_rule (according to N (scaling factor) when needed)

    Input arguments:
    - w_val  : original weight value passed by pars_conn to be scaled
    - w_rule : weight rule to scale w_val (if None just provide w_val as given)
    - N      : (only if w_rule is specified) divide w_val by some function of N

    Return:
    - rescaled w_val according to w_rule

    NOTE: if N==0 then return a zero weight.
    """
    # Treatment of scaling
    if w_rule == '1/sqrt_n':
        wv = np.divide(w_val, np.sqrt(N), out=np.zeros_like(w_val), where=N>0)
    elif w_rule == '1/n':
        wv = np.divide(w_val, N, out=np.zeros_like(w_val), where=N>0)

    # Output handling
    if w_rule in ['1/sqrt_n','1/n']:
        return wv
    else:
        # If none of the w_rules applies, then just return w_val value (as passed to function)
        return w_val

# -----------------------------------------------------------------------------------------------------------------------
# ROUTINES FOR NETWORK PARAMETERS
# -----------------------------------------------------------------------------------------------------------------------
def nea_parameters(N_e, N_i, N_g, connectivity=True,
        nea_setup='default',
        topology={}, p_conn={}, w_scaling={},
        spatial_geometry=None,
        ne_pars={}, ni_pars='same', ng_pars={}, ecs_pars={},
        ee_pars='default', ie_pars='same', ii_pars='default', ei_pars='same',
        randomize_ics={},
        **kwargs):
    """
    Generate parameter dictionaries for neuron-glia networks (does not include external stimulation parameters).

    Input arguments:
    - N_e   : Number of E neurons
    - N_i   : Number of I neurons
    - N_g   : Number of astrocytes
    - connectivity : {True} | False | dict Produce connectivity matrices and edge information (may be slow for large networks).
                     Set it to False when you need to retrieve only parameters but not build the network as in the MF analysis,
                     Provide instead a dictionary {'C_conn':..., 'N_conn':..., 'edges':..., 'syn_edges':...} for fixed connectivity.
    - ngn_setup : {'default'}|{'network'}|'single-astro'  Predefined models: Network vs. single neuron/astrocyte with N synapses
    - topology  : {} | {'ee': string, 'ie': ... , 'ii': ..., 'ei': ...., 'gg': ...} String: {'random'}|{'all-to-one'} (see connect()).
    - p_conn    : {'ee': pval, 'ie': pval, 'ii': pval, 'ei': pval, 'gg': pval, 'gee': ... 'eeg': ...} : Connection probabilities (must be pval \in [0,1])
                  If left empty, p_conn are automatically set according to N_x/C_x as described in van Vreswijk and Sompolinsky, 1998 or Brunel JCN 2000.
    - w_scaling : {'key': rule_string | float } : Rule for weight scaling or float value.
                  Keys  : 'jee','jie','jei','jii','jgg','wee','wie','wei','wii'
                  Rules : '1/sqrt_n','1/n'.
                  NOTE : to set 'wsg' (gliotransmitter connection" you need to do it through pars_glt={'js': ...}
    - ne_pars   : LIF parameters for E neurons
    - ni_pars   : LIF parameters for I neurons
    - ng_pars: LIF parameters for astrocytes
    - glt_pars  : Gliotransmitter parameters (see gliot_params)
    - ee_pars   : EE synapse parameters (see mhv_params)
    - ie_pars   : IE synapse parameters (see mhv_params) | {'same'} If 'same', set IE pars identical to EE ones
    - ii_pars   : IE synapse parameters (see mhv_params)
    - ei_pars   : II synapse parameters (see mhv_params) | {'same'} If 'same', set EI pars identical to II ones
    - prer_ee_pars  : Presynaptic receptor parameters for EE synapses (see prer_params)
    - prer_ie_pars  : Presynaptic receptor parameters for EE synapses (see prer_params) | {'same'} If 'same', set IE pars identical to EE ones
    - prer_ii_pars  : Presynaptic receptor parameters for EE synapses (see prer_params)
    - prer_ei_pars  : Presynaptic receptor parameters for EE synapses (see prer_params) | {'same'} If 'same', set EI pars identical to II ones
    - **kwargs:
      - Connection weight specifics: 'jee','jie','jei','jii','jgg','wee','wie','wei','wii','C','A'

    Returns
    - pars_lif : LIF parameter dictionaries organized by ['e','i','g'] keys.
    - pars_mhv : MHV synapse parameter dictionaries organized by ['ee','ie','ei','ii'] keys.
    - pars_glt : gliotransmitter parameter dictionary (identical for all connections).
    - pars_prer: Presynaptic parameter dictionaries organized by ['ee','ie','ei','ii'] keys.
    - pars_conn: Connection weights' dictionary.
    - C_conn   : Connection matrix dictionary organized by ['ee','ie','ii','ei','gg','ng','gn'] keys.
    - N_conn   : Connection numbers dictionary organized by ['ee','ie','ii','ei','gg','ng','gn'] keys.
    - edges    : Edges dictionary organized by ['ee','ie','ii','ei','gg','ng','gn'] keys.
    - syn_edges: Dictionary of connection edges by synapses organized by the following keys:
                 ['ee_s','ie_s','ii_s','ei_s', # Neuron-to-synapse
                  's_ee','s_ie','s_ii','s_ei', # Synapse-to-neuron
                  'ee_g','ie_g','ii_g','ei_g', # xx-synapse-to-astrocyte
                  'g_ee','g_ie','g_ii','g_ei'] # astrocyte-to-xx-synapse

    v1.0
    Maurizio De Pitta', The University of Chicago, Chicago, May 1st, 2016.
    """

    # -----------------------------------------------------------------------------------------------------------------------
    # Default Parameters
    # -----------------------------------------------------------------------------------------------------------------------
    # Allocate dictionaries for parameters
    pars_cell_default, pars_syn_default, pars_ecs_default = {}, {}, {}

    # Default cell parameters
    pars_cell_default['e'] = neuron_parameters('l5e')
    pars_cell_default['i'] = neuron_parameters('l5i')
    pars_cell_default['g'] = astrocyte_parameters()

    # Default ECS parameters
    pars_ecs_default = ecs_parameters()

    # Default Synaptic connections
    pars_syn_default['ee'] = synapse_parameters(ttype='glu')
    pars_syn_default['ie'] = synapse_parameters(ttype='glu')
    pars_syn_default['ei'] = synapse_parameters(ttype='gaba')
    pars_syn_default['ii'] = synapse_parameters(ttype='gaba')

    # TODO: --------------------Define Connectivity
    # Synaptic weight for connections (astro-to-syn connections are defined instead in gliot_parameters)
    # IMPORTANT: pars_conn will be finalized however at the end of the routine
    pars_conn = {'C': 0,  # Number of connections per neuron
        'G': 0,  # Number of connections per astrocyte
        'K': 0,
        # Number of total connections (unless given, it is computed internally once connections are created)
        'K_ee_v': 0,  # Vector of number of synapses/astrocyte for normalization of EE weights
        'K_ie_v': 0,  # Vector of number of synapses/astrocyte for normalization of IE weights
        'K_ei_v': 0,  # Vector of number of synapses/astrocyte for normalization of EI weights
        'K_ii_v': 0,  # Vector of number of synapses/astrocyte for normalization of II weights
        'K_ee': 0,  # Vector of number of EE synapses/astrocyte
        'K_ie': 0,  # Vector of number of IE synapses/astrocyte
        'K_ei': 0,  # Vector of number of EI synapses/astrocyte
        'K_ii': 0,  # Vector of number of II synapses/astrocyte
        'jee': 1.0,  # J_EE connections
        'jie': 1.0,  # J_IE connections
        'jei': 1.0,  # J_EI connections
        'jii': 1.0,  # J_II connections
        'jgg': 1.0,  # J_GG connections
        'wee': 1.0,  # W_EE connections
        'wie': 1.0,  # W_IE connections
        'wei': 1.0,  # W_EI connections
        'wii': 1.0,  # W_II connections
        'wsg': None,  # --> pars_glt['js] (added later)
        'cj': 1.0,
        # Scaling factor for synaptic connections (used by bifurcation analysis, see Stability.Analysis)
        'cw': 1.0,
        # Scaling factor for syn-astro connections (used by bifurcation analysis, see Stability.Analysis)
        'cx': 1.0,
        # Scaling factor for external const input to neurons (used by bifurcation analysis, see Stability.Analysis)
        'cs': 1.0,
        # Scaling factor for s.d. external input to neurons (used by bifurcation analysis, see Stability.Analysis)
        'gj': 1.0,  # Scaling factor for inhibitory synaptic weights (assuming J_EE = J_IE = J)
        'ge': 1.0,  # Scaling factor for J_EI = g_E * J_EE
        'gi': 1.0,  # Scaling factor for J_II = g_I * J_IE
        'fxi': 1.0,  # Scaling factor for xi_IE = f_xi * xi_EE (varies in [0,1/xi_EE])
    }
    # Custom parameters
    pars_conn = gu.varargin(pars_conn, **kwargs)

    # Default weights' scaling rules (see set_weights() for rules)
    weights = {'jee': '1/sqrt_n',  # J_EE connections
        'jie': '1/sqrt_n',  # J_IE connections
        'jei': '1/sqrt_n',  # J_EI connections
        'jii': '1/sqrt_n',  # J_II connections
        'jgg': '1/n',  # J_GG connections
        'wee': '1/sqrt_n',  # W_EE connections
        'wie': '1/sqrt_n',  # W_IE connections
        'wei': '1/sqrt_n',  # W_EI connections
        'wii': '1/sqrt_n',  # W_II connections
    }
    # Custom scaling rules
    weights = gu.varargin(weights, **w_scaling)
    # TODO: --------------------End of TODO for Connectivity

    # Default connections pathways
    keys_conn = ['ee', 'ie', 'ii', 'ei', 'gg', 'gee', 'gie', 'gii', 'gei', 'eeg', 'ieg', 'iig', 'eig']
    # Default values
    # TODO: --------------------Begin of TODO for Topology
    if (nea_setup == 'model-A') or (nea_setup == 'model-B'):
        # Model A in Brunel, JCN 2000
        network_topology = dict(list(zip(keys_conn,['random-const-outdeg']*4 + [None] + ['s2g-random-const-outdeg']*4 + ['fixed']*4)))  # All connections are randomly picked (except gliotransmitter ones)
        # In this configuration the connectivity is provided by the user through 'ee'. 'gg' is initialized to 0. for completeness. glt_conn are set to 1. only for excitatory synapses.
        prob_conn = dict(list(zip(keys_conn, [1. if k not in ['gg','eig','iig'] else 0. for k in keys_conn])))
    elif (nea_setup == 'single-astro'):
        network_topology = dict(list(zip(keys_conn, ['all-to-one'] + [None]*12)))
        prob_conn = dict(list(zip(keys_conn, [1. if 'ee' in k else 0. for k in keys_conn])))
    elif (nea_setup == 'l5e+ecs'):
        network_topology = dict(list(zip(keys_conn, ['all-to-one'] + [None]*12)))
        prob_conn = dict(list(zip(keys_conn,[1. if k=='ee' else 0. for k in keys_conn])))
    # Handling of the probability
    prob_conn = gu.varargin(prob_conn,**p_conn)
    # TODO: --------------------End of TODO for Topology

    # Geometry parameters and clustering
    # TODO: CLSUTERING: the code offer limited control on clustering: NGN are in correlated clusters: EI cannot be clustered separately w.r.t. glia
    # We specify first any geometric quantity, regardless (this dictionary will be used only if spatial_geometry!=None
    network_geometry = geometry_parameters(N_e,N_i,N_g,spatial_geometry, **kwargs)

    # Create default parameters -- cluster compatible
    # This block is going to create 'C_e' and 'C_i' which are then used to infer connection weights
    if (nea_setup == 'model-A') or (nea_setup == 'model-B'):
        assert (np.size(list(p_conn.keys())) >= 1), "model-A and model-B setup require to specify one prob value for all connections (either as 'ee','ie','ei',ii')"
        if (network_geometry['N_clusters']>1)and(np.size(prob_conn['ee']==1)):
            # This is the case where we have clustering by prob_conn['ee] is a scalar: it extends to the matrix level
            prob_conn['ee'] *= np.ones((network_geometry['N_clusters'],network_geometry['N_clusters']))
        for k in ['ie','ei','ii']: prob_conn[k] = prob_conn['ee']

    # TODO: Cluster configuration (NOT IMPLEMENTED for NEANs)
    # # Make probabilities suitable for clustered network if detected
    # if network_geometry['N_clusters']>1:
    #     for k in ['ee','ie','ei','ii','gg']:
    #         if (np.size(prob_conn[k])==1)and(prob_conn[k]>0):
    #             prob_conn[k] *= np.ones((network_geometry['N_clusters'],network_geometry['N_clusters']))
    #         else:
    #             # Check if probabilities are given as matrices with correct shape
    #             assert all([n==network_geometry['N_clusters'] for n in np.shape(prob_conn[k])]),'prob_conn['+k+'] must be a matrix of size N_clusters*N_clusters or a scalar'

    # Topology Custom parameters
    network_topology = gu.varargin(network_topology,**topology)
    # Force network topology in the case of null-sets
    if N_e <= 0: network_topology.update({'ee':None,'ie':None,'gee':None,'gie':None,'eeg':None,'ieg':None})
    if N_i <= 0: network_topology.update({'ii':None,'ei':None,'gii':None,'gei':None,'iig':None,'eig':None})
    if N_g <= 0: network_topology.update({'gg':None,'gee':None,'gie':None,'gii':None,'gei':None,'eeg':None,'ieg':None,'iig':None,'eig':None})
    # Update probability of connections
    for k,v in network_topology.items():
        if v == None: prob_conn[k] = 0.0

    # Custom ICs
    # ICs defaults
    rand_ics = {'e': False, 'i': False, 'g': False}  # Not randomized (i.e. fixed) ICs
    rand_ics = gu.varargin(rand_ics,**randomize_ics)

    # TODO: ------------- START Connectivity Settings -- the clustering mode should be disabled -- the rest should be fine
    # Set connectivity based on probability
    if (nea_setup == 'model-A') or (nea_setup == 'model-B'):
        try:
            # If clustering is present, prob_conn['ee'] will be a matrix and the following will run
            pars_conn['C_e'] = (prob_conn['ee'].T @ network_geometry['N_e_cluster']).astype(np.int32)  # Use matrix product on transpose to get all afferents connections per cluster
            pars_conn['C_i'] = (pars_conn['C_e'] * network_geometry['N_i_cluster']/network_geometry['N_e_cluster']).astype(np.int32)
        except:
            # If there is no clustering it will resort to scalar computation
            pars_conn['C_e'] = int(prob_conn['ee']*N_e)
            pars_conn['C_i'] = int(pars_conn['C_e']*N_i/N_e)
        # Do the same for the glia network if connections are detected, unless C_g is externally specified in this case
        if ('C_g' not in list(kwargs.keys())):
            try:
                pars_conn['C_g'] = (prob_conn['gg'].T @ network_geometry['N_g_cluster']).astype(np.int32)
            except:
                pars_conn['C_g'] = int(prob_conn['gg']*N_g)
        else:
            # If external glial connections are specified externally, make sure that the number is an integer
            pars_conn['C_g'] = int(kwargs['C_g'])
    else:
        # TODO: update this case by correct assignment
        print("Warning: Synaptic weight not properly assigned -- MUST be specified manually (TODO in future versions)")
        pars_conn['C_e'] = prob_conn['ee']*N_e
        pars_conn['C_i'] = prob_conn['ii']*N_i
        pars_conn['C_g'] = prob_conn['gg']*N_g

    if 'edges' in kwargs.keys():
        given_edges = kwargs['edges']
    else:
        given_edges = {}
    # If connectivity is False, the following provides default (demi-empty) dicts
    # TODO: Link to connectivity (conn) module
    N_conn, edges, coords = conn.ngn_connections(N_e, N_i, N_g,
        connectivity=connectivity,
        p_conn=prob_conn,
        topology=network_topology,
        spatial_geometry=network_geometry,
        given_edges=given_edges)
    # TODO: ------------- END Connectivity Settings

    # Allocate population specific parameters
    pars_cell, pars_syn, pars_ast = {}, {}, {}

    # Neuron parameters
    pars_cell['e'] = set_module_parameters(module_pars=ne_pars, module_default=pars_cell_default['e'],module_same=pars_cell_default['e'])
    if nea_setup == 'model-B':
        pars_cell['i'] = set_module_parameters(module_pars=ni_pars, module_default=pars_cell_default['i'],module_same=pars_cell_default['i'])
    else:
        pars_cell['i'] = set_module_parameters(module_pars=ni_pars, module_default=pars_cell_default['i'],module_same=pars_cell['e'])
    # Astro parameters
    pars_cell['g'] = set_module_parameters(module_pars=ng_pars, module_default=pars_cell_default['g'],module_same=pars_cell_default['g'])
    # ECS parameters
    # Astro parameters
    pars_ecs = set_module_parameters(module_pars=ecs_pars, module_default=pars_ecs_default,module_same=pars_ecs_default)
    # Synapse parameters
    pars_syn['ee'] = set_module_parameters(module_pars=ee_pars, module_default=pars_syn_default['ee'],module_same=pars_syn_default['ee'])
    if nea_setup == 'model-B':
        pars_syn['ie'] = set_module_parameters(module_pars=ie_pars, module_default=pars_syn_default['ie'],module_same=pars_syn_default['ie'])
        pars_syn['ei'] = set_module_parameters(module_pars='same', module_same=pars_syn['ee'])
        pars_syn['ii'] = set_module_parameters(module_pars='same', module_same=pars_syn['ie'])
    else:
        pars_syn['ii'] = set_module_parameters(module_pars=ii_pars, module_default=pars_syn_default['ii'],module_same=pars_syn_default['ii'])
        pars_syn['ie'] = set_module_parameters(module_pars=ie_pars, module_default=pars_syn_default['ie'],module_same=pars_syn['ee'])
        pars_syn['ei'] = set_module_parameters(module_pars=ei_pars, module_default=pars_syn_default['ei'],module_same=pars_syn['ii'])

    # NOTE: In the NEA setup the weights are only taken proportional to the normalized depolarization scale
    if ('jee' not in list(kwargs.keys())): pars_conn['jee'] = set_weights(1.0,weights['jee'], pars_conn['C_e'])
    if ('jie' not in list(kwargs.keys())): pars_conn['jie'] = set_weights(1.0,weights['jie'], pars_conn['C_e'])
    if ('jei' not in list(kwargs.keys())): pars_conn['jei'] = set_weights(1.0,weights['jei'], pars_conn['C_i'])
    if ('jii' not in list(kwargs.keys())): pars_conn['jii'] = set_weights(1.0,weights['jii'], pars_conn['C_i'])

    # TODO: ------------- START: Revise glia network weights
    # Set interconnecting glial weights (also taken proportional to an arbitrary normalized depolarization scale
    # Later we will check If C_g!=G and rescale J_gg accordingly. This is actually redundant
    if ('jgg' not in list(kwargs.keys())): pars_conn['jgg'] = set_weights(1.0,weights['jgg'], pars_conn['C_g'])
    # TODO: In a clustered network this should be provided as a vector of size 1xN_clusters in future MFA implementations that might deal with clustered networks
    pars_conn['J_gg'] = np.mean(pars_conn['jgg'],dtype=int)   # Create a mean value needed for the computation of moments in ICs reassignment for the spiking neuron
    # TODO: ------------- END: Set glia network weight

    # TODO: ------------- START: Connectivity update: Spatial Connectivity is crucial here
    # The following is needed to properly set synaptic weights
    # The true number of connections
    if connectivity:
        if N_g > 1:
            # The following is needed to have the same treatment for no-cluster and clustered network
            if network_geometry['N_clusters'] < 2:
                glia_id = [np.arange(N_g)]
                pars_conn['K'] = [pars_conn['K']] # Wrap it as list for compatibility
            else:
                glia_id = network_geometry['g_cluster']
                pars_conn['K'] = [pars_conn['K']] * network_geometry['N_clusters']
            # K computation
            if ('K' not in list(kwargs.keys())):
                for ci,gid in enumerate(glia_id):
                    # Get how many synapses per glia cell within each cluster we have
                    _, pars_conn['K'][ci] = np.unique(np.r_[edges['gee'][-1][np.isin(edges['gee'][-1],gid)],
                    edges['gie'][-1][np.isin(edges['gie'][-1],gid)],
                    edges['gei'][-1][np.isin(edges['gei'][-1],gid)],
                    edges['gii'][-1][np.isin(edges['gii'][-1],gid)]], return_counts=True)
            # Create per-synapse values of number of connections for the associated astrocyte -- These K_vals represent the normalization terms for all synapses i.e.
            K_vals = {}  # Temporary dictionary whereto dump counts per synapse type
            for k in ['ee','ei','ie','ii']:
                K_vals[k] = np.ones(np.shape(edges['g' + k])[-1])
                glia_idx,counts = np.unique(edges['g' + k][-1],return_counts=True)
                for ev,cnt in zip(glia_idx,counts):
                    # K_vals[k][edges['g' + k][-1] == ev] = np.count_nonzero(edges['g' + k][-1] == ev)
                    K_vals[k][edges['g' + k][-1]==ev] = cnt
                pars_conn['K_' + k + '_v'] = K_vals[k].astype(dtype=np.uint32)  # Retrieve vectors of all synapses numbers to normalize individual synapses per type
                pars_conn['K_' + k] = np.atleast_1d([np.mean(pars_conn['K_'+k+'_v'][np.isin(edges['g'+k][-1],gid)],dtype=np.int32).item() for gid in glia_id])  # Retrieve mean number of connections per type per astrocyte cell
            # Compute K as mean per cluster
            pars_conn['K'] = np.atleast_1d([np.mean(K,dtype=np.int32) for K in pars_conn['K']])
            # Handling of the non-clustered case requires just scalars
            # NOTE: in the spiking network not necessarily K = \sum K_xx since the K_xx due to rounding approximations
            if network_geometry['N_clusters'] < 2:
                # If N_clusters <2 K and K_xx are provided as scalar (this assures back-compatibility and is also used in the MFA routines at the moment)
                pars_conn['K'] = pars_conn['K'].item(0)
                for k in ['ee','ei','ie','ii']:
                    pars_conn['K_' + k] = pars_conn['K_' + k].item(0)

            if spatial_geometry!=None:
                # Produce min and max number of connections for later use
                # (the min/max number of connections is approximated -- may not reflect the min/max number of connections of an astrocyte of the network)
                # min and max here are theoretical extrema taken by the sum of min/max values of individual K_xx_v
                pars_conn['K_min'] = np.sum([np.min(pars_conn['K_'+k+'_v']) for k in ['ee','ei','ie','ii']])
                pars_conn['K_max'] = np.sum([np.max(pars_conn['K_'+k+'_v']) for k in ['ee','ei','ie','ii']])

            # G computation
            if np.any(prob_conn['gg']>0.):
                if network_geometry['N_clusters'] < 2:
                    glia_id = [np.arange(N_g)]
                    pars_conn['G'] = [pars_conn['G']] # Wrap it as list for compatibility
                else:
                    glia_id = network_geometry['g_cluster']
                    pars_conn['G'] = [pars_conn['G']] * network_geometry['N_clusters']
                if ('G' not in list(kwargs.keys())):
                    for ci,gid in enumerate(glia_id):
                        # Get how many synapses per glia cell within each cluster we have
                        _, pars_conn['G'][ci] = np.unique(edges['gg'][-1][np.isin(edges['gg'][-1],gid)], return_counts=True)
                # Compute G as mean per cluster
                pars_conn['G'] = np.atleast_1d([np.mean(G,dtype=np.int32) for G in pars_conn['G']])
                # Handling of the non-clustered case requires just scalars
                if network_geometry['N_clusters'] < 2:
                    # If N_clusters <2 K and K_xx are provided as scalar (this assures back-compatibility and is also used in the MFA routines at the moment)
                    pars_conn['G'] = pars_conn['G'].item(0)
    else:
        # TODO: Not tested for clustered networks // There might be some requirements in the size of the K when passed to a clustered network that are not considered here
        # This case overruns the situation where connectivity is not computed.
        # In this case it just checks whether K_xx are passed as additional parameters
        for k in ['ee', 'ei', 'ie', 'ii']:
            if ('K_'+k+'_v' in list(kwargs.keys())): pars_conn['K_'+k+'_v'] = kwargs['K_'+k+'_v']
            if ('K_'+k in list(kwargs.keys())): pars_conn['K_'+k] = kwargs['K_'+k]
        K_vals = np.r_[pars_conn['K_ee'], pars_conn['K_ie'], pars_conn['K_ei'], pars_conn['K_ii']]
        if None not in K_vals:
            pars_conn['K'] = np.round(np.sum(K_vals)).item()
        else:
            pars_conn['K'] = 0
    #TODO: ------------- END: Connectivity update

    # If G and c_g differ the following rescales jgg by G rather than C_g. This is because extsynstim_paramaeters later uses 'G' for computation of ratet and not 'C_g'
    # NOTE: All the weights are taken proportional to normalized depolarization scale
    # TODO: you should fix K and G based on the topology of the network and compute them by formulas, rather than making them realization dependent -- The only way to bypass this at the moment is assigning J_gg and W_xx
    if ('jgg' not in kwargs.keys())and(any(np.atleast_1d(np.abs(pars_conn['G']-pars_conn['C_g'])))):
        pars_conn['jgg'] = set_weights(1.0,weights['jgg'],pars_conn['G'])
        pars_conn['J_gg'] = np.mean(pars_conn['jgg'],dtype=int)
    for k in ['ee','ie','ei','ii']:
        if spatial_geometry==None:
            if ('w'+k not in list(kwargs.keys())): pars_conn['w'+k] = set_weights(np.ones(np.size(pars_conn['K_'+k+'_v'])),weights['w'+k],pars_conn['K_'+k+'_v'])
        else:
            if ('w'+k not in list(kwargs.keys())): pars_conn['w' + k] = set_weights(1.0,weights['w'+k],np.mean([pars_conn['K_min'],pars_conn['K_max']]))

    # TODO: ------------- START: Refine weights based on E-I balance
    if nea_setup == 'model-A':
        pars_conn['jie'] = pars_conn['jee']
        pars_conn['jii'] = pars_conn['jee']*pars_conn['gj']
        pars_conn['jei'] = pars_conn['jee']*pars_conn['gj']
    elif nea_setup == 'model-B':
        pars_conn['jei'] = pars_conn['ge']*pars_conn['jee']
        pars_conn['jii'] = pars_conn['gi']*pars_conn['jie']

    for k in ['ee','ie','ei', 'ii']:
        pars_conn['w'+k] *= pars_conn['cw']
        # Generate entries as 'W_xy' for mean values of connection weight vectors 'wxy'
        pars_conn['W_'+k] = np.mean(pars_conn['w'+k])
    # TODO: ------------- END: Refine weights based on E-I balance

    # TODO: ------------- START: Randomize ICs
    # # NOTE: this option is currently used to avoid synch firing when the network is fed by const input to all neurons
    # for k, v in rand_ics.items():
    #     if k == 'e':
    #         N_ = N_e
    #     elif k == 'i':
    #         N_ = N_i
    #     else:
    #         N_ = N_g
    #     if v:
    #         # pars_lif[k]['ICs'] = (pars_lif[k]['vr']-pars_lif[k]['vl'])+1.05*(pars_lif[k]['vt']-pars_lif[k]['vr'])*np.random.random((N_))
    #         pars_lif[k]['ICs'] = (pars_lif[k]['vr'] - pars_lif[k]['vl']) + (pars_lif[k]['vt'] - pars_lif[k]['vr'])*np.random.random((N_))
    #     else:
    #         pars_lif[k]['ICs'] = pars_lif[k]['vr']*np.ones(N_)
    # TODO: ------------- END: Randomize ICs

    # Return update parameters
    # return pars_lif, pars_syn, pars_glt, pars_prer, pars_conn, C_conn, N_conn, edges, syn_edges
    return pars_cell, pars_syn, pars_ecs, pars_conn, N_conn, edges, coords, network_geometry

def extstim_parameters(N, pars_lif, N_clusters=0, ex_topology='default', external=False,
        ex_pars={}, ix_pars='same', gx_pars={},
        prer_ex_pars={}, prer_ix_pars='same', prer_gx_pars={},
        **kwargs):
    """
    Produce data for external spiking stimulation.

    Input arguments:
    - N        : Dictionary with keys 'e','i','g' with values of corresponding node numbers.
    - twin     : 1x2 Array of floats for time window of simulation.
    - ex_pars  : Parameters for X-->E neurons inputs   : {'default'} | {}
    - ix_pars  : Parameters for X-->I neurons inputs   : 'default' | {'same'} | {}
    - gx_pars  : Parameters for X-->G astrocyte inputs : {'default'} | {}
    See extsyn_paramaters() for 'default' parameters as well as syn.mhv_parameters() and prer_parameters().
    - **kwargs :
      - Connectivity parameters: wex,wix,wgx,wgex,wgix,wggx

    Return:
    - pars_extsyn   : Dictionary of parameter dictionaries for incoming external synapses.
    - pars_prer_ext : Dictionary of parameter dictionaries for gliotransmitter regulation of external synapses.
    - ext_conn : Dictionary of external connection edges.
    - spk      : Dictionary of 3*NSPK arrays of float such that: [t_spk; syn_spike_index; input_neuron_index]

    Maurizio De Pitta', The University of Chicago, May 29th, 2016.
    """

    mux = lambda C, w, nu: C*w*nu

    # Default parameters for number of E,I,G cells to be considered for external synapses
    N_cell_default = {'e': 0,
                      'i': 0,
                      'g': 0}
    # Default parameters for external synapses
    pars_extsyn_default = {'ex': extsyn_parameters(),
                           'ix': extsyn_parameters(),
                           'gx': extsyn_parameters()}
    # Default parameters for gliotransmitter modulation of external synapses
    pars_prer_ext_default = {'ex': prer_parameters(),
                             'ix': prer_parameters(),
                             'gx': prer_parameters()}
    # Spike timing
    # VERY IMPORTANT: this data type changes with "external" value:
    # If external=True, spk provides the spike train; if external=False, spk is a dict of dictionary with entries 'rate' and 'T_rate'
    if external:
        spk = {'ex': np.zeros((2, 0), dtype=float),
               'ix': np.zeros((2, 0), dtype=float),
               'gx': np.zeros((2, 0), dtype=float)}
    else:
        spk = {}
    # Connectivity
    ext_conn = {'ex_s': np.zeros((2, 0), dtype=np.intc),  # External Poisson neuron to external synapses to E neurons
                'ix_s': np.zeros((2, 0), dtype=np.intc),  # External Poisson neuron to external synapses to I neurons
                'gx_s': np.zeros((2, 0), dtype=np.intc),  # External Poisson neuron to external synapses to G cells
                's_ex': np.zeros((2, 0), dtype=np.intc),  # External synapses to E neurons
                's_ix': np.zeros((2, 0), dtype=np.intc),  # External synapses to I neurons
                's_gx': np.zeros((2, 0), dtype=np.intc),  # External synapses to G cells
                'ex_g': np.zeros((2, 0), dtype=np.intc),  # External EX synapses to G cells
                'ix_g': np.zeros((2, 0), dtype=np.intc),  # External IX synapses to G cells
                'gx_g': np.zeros((2, 0), dtype=np.intc),  # External GX synapses to G cells
                'g_ex': np.zeros((2, 0), dtype=np.intc),  # G cells to External EX synapses
                'g_ix': np.zeros((2, 0), dtype=np.intc),  # G cells to External IX synapses
                'g_gx': np.zeros((2, 0), dtype=np.intc)   # G cells to External GX synapses
                }  # External synapses to G cells
    pars_conn_ext = {'jex': 0.0,  # W_EX external connections
                     'jix': 0.0,  # W_IX external connections
                     'jgx': 0.0,  # W_GX external connections (from GX to G as "wggx")
                     'wgex': 0.0,  # W_GEX external connections (from EX to G)
                     'wgix': 0.0,  # W_GIX external connections (from IX to G)
                     # 'wggx': 0.0,    # W_GGX external connections (from GX to G)
                     'wexg': 0.0,  # W_EXG external connections (from G to EX)
                     'wixg': 0.0,  # W_IXG external connections (from G to IX)
                     'wgxg': 0.0  # W_GXG external connections (from G to GX)
                     }
    weights = {'jex': '1/sqrt_n',  # W_EX external connections
               'jix': '1/sqrt_n',  # W_IX external connections
               'jgx': '1/n'  # W_GX external connections
               }

    # Custom number of E,I,G cells targeted by external inputs
    N = gu.merge_dicts(N_cell_default, N)
    for k, v in N.items(): N[k] = int(v)

    # Custom external connections
    pars_extsyn = {}
    pars_extsyn['ex'] = set_module_parameters(module_pars=ex_pars, module_default=pars_extsyn_default['ex'],module_same=pars_extsyn_default['ex'])
    pars_extsyn['ix'] = set_module_parameters(module_pars=ix_pars, module_default=pars_extsyn_default['ix'],module_same=pars_extsyn['ex'])
    pars_extsyn['gx'] = set_module_parameters(module_pars=gx_pars, module_default=pars_extsyn_default['gx'],module_same=pars_extsyn_default['gx'])
    # Check consistency of rho with number of clusters: if rho is specified as a vector (consistent with clustered network) but the size is not N_clusters
    # then halts execution and issue an error
    for k in ['e','i','g']:
        if np.size(pars_extsyn[k+'x']['rho'])>1:
            assert np.size(pars_extsyn[k+'x']['rho'])==int(N_clusters),'Size of rho in '+k.upper()+'X group must be equal to N_clusters'

    # Set stimulus specifics for each LIF group
    for k, px in pars_extsyn.items():
        if px['rate']==None:
            # If rate is None, we do not need to set all the spikes and the vectors. This is the case in most of the
            # situation (in the C++ code instead we were passing the whole spikes).
            continue
        if hasattr(px['rate'],'__len__'):
            px['rate'] = np.atleast_1d(px['rate'])
        if np.any(px['N_s']) and np.any(px['rate']) and (N[k[0]] > 0):
            if external:
                spk = None
                pass
                # DISABLED
                # # if True then spikes are fed into the model according to whatever timing is given by the user
                # # if False skips it and will generate spikes internally by Poisson generators in the C/C++ code
                # spk[k] = spg.input_spikes(px['N_s']*N[k[0]], px['T_rate'], px['rate'], px['trpn'],
                #     stimulus=px['stimulus'], spikes_pre=px['spikes_pre'])
                # # Round t_spk to decimal PREC
                # if spk[k].size > 0: spk[k][0] = np.around(spk[k][0], PREC)
            # Establish connections with neurons
            if ex_topology == 'default':
                # Default connectivity is all-to-all: Say you have M external neurons, and N target neurons, the
                # connection matrix will be: [[0,0,...,0,  1,1,...,1,  2,2,....,M-2,M-1,M-1....,M-1];
                #                             [0,1,...,N-1,0,1,...,N-1,0,1,..., N-1,0  ,1,...., N-1]]
                # Synapses are counted from [0,...N*M-1] and associated to first and last row above accordingly.
                ext_index = np.concatenate([i*np.ones(N[k[0]]) for i in range(px['N_s'])])  # Indexes of external neurons
                syn_index = np.arange(px['N_s']*N[k[0]])  # Total number of synapses
                neu_index = np.concatenate([np.arange(N[k[0]]) for i in range(px['N_s'])])
                print("WARNING: external=False is not equivalent to external=True in the current implementation!!!")
                if not external:
                    # In case of internal Poisson neurons assumes that by default each external synapse associates with
                    # one neuron
                    # NOTE: The corresponding C-code also in principle contains data structures that could allow for much more
                    # variegated connections like neurons that sends connections to different cell types. But this is not
                    # considered at the moment,
                    ext_conn[k+'_s'] = (np.vstack((ext_index, syn_index))).astype(np.intc)  # intc type must be passed as integers to C routines
                # TODO: There will be a mismatch in case external inputs will be provided, insofar as we are assuming that
                # TODO: M external neurons are all connected to N neurons of the network, so you have M inputs only that goes to all the N neurons.
                # TODO: The current configuration instead with external=False assumes instead that you have N*M independent inputs (there is a mismatch
                ext_conn['s_'+k] = (np.vstack((syn_index, neu_index))).astype(np.intc)  # intc type must be passed as integers to C routines
                # # Assign each external synapse to one of the neurons in N[k] and append as a third line to spk sequence
                # spk[k] = (np.vstack((spk[k],np.mod(spk[k][1],N[k])))).astype(float)
                # # Retrieve external connectivity (the Set instance is used to get unique tuples)
                # ext_conn[k] = np.array(zip(*tuple(sets.Set(map(tuple,spk[k][1:].T)))),dtype=np.intc)
            else:
                # WARNING: This is not checked: It assumes that ex_topology is just given as a dictionary of the same keys
                if not external:
                    ext_conn[k+'_s'] = ex_topology[k+'s_'].astype(np.intc)
                ext_conn['s_'+k] = ex_topology['s_'+k].astype(np.intc)
            # Connections with glial cells (set only if glia cells are present)
            if N['g'] > 0:
                # NOTE: Present version only considers bidirectional connections for syn_frac fraction of synapses with
                # the first glial cell
                if k in ['ex', 'ix']:
                    assert px['syn_frac'] <= 1.0, "syn_frac must be in [0,1] for group "+k
                    if px['syn_frac'] > 0:
                        N_g = int(np.floor(px['syn_frac']*(np.amax(ext_conn['s_'+k][0]+1))))
                        ext_conn[k+'_g'] = (np.vstack((np.arange(N_g), np.zeros(N_g)))).astype(np.intc)
                        # For some weird reason flipping the array is not equivalent in C
                        ext_conn['g_'+k] = (np.vstack((ext_conn[k+'_g'][1], ext_conn[k+'_g'][0]))).astype(np.intc)
                else:
                    px['syn_frac'] = 1.0  # All external synapses to glia cells must impinge on glia
                    # For some weird reason flipping the array is not equivalent in C
                    ext_conn['g_'+k] = (np.vstack((ext_conn['s_'+k][1], ext_conn['s_'+k][0]))).astype(np.intc)

    # Set parameters for gliotransmitter regulation of external synaptic contacts
    pars_prer_ext = {}
    pars_prer_ext['ex'] = set_module_parameters(module_pars=prer_ex_pars,module_default=pars_prer_ext_default['ex'],module_same=pars_prer_ext_default['ex'])
    pars_prer_ext['ix'] = set_module_parameters(module_pars=prer_ix_pars,module_default=pars_prer_ext_default['ix'],module_same=pars_prer_ext['ex'])
    pars_prer_ext['gx'] = set_module_parameters(module_pars=prer_gx_pars,module_default=pars_prer_ext_default['gx'],module_same=pars_prer_ext_default['gx'])

    # Set connection weights
    # Update with external values
    for k, v in pars_conn_ext.items():
        if k in kwargs:
            pars_conn_ext[k] = kwargs[k]
    # Special scaling for specific connections
    pars_conn_ext['jex'] = set_weights(pars_lif['e']['vt']-pars_lif['e']['vr'], weights['jex'],pars_extsyn['ex']['N_s']) if ('jex' not in list(kwargs.keys())) else kwargs['jex']
    pars_conn_ext['jix'] = set_weights(pars_lif['i']['vt']-pars_lif['i']['vr'], weights['jix'],pars_extsyn['ix']['N_s']) if ('jix' not in list(kwargs.keys())) else kwargs['jix']
    pars_conn_ext['jgx'] = set_weights(pars_lif['g']['vt']-pars_lif['g']['vr'], weights['jgx'],pars_extsyn['gx']['N_s']) if ('jgx' not in list(kwargs.keys())) else kwargs['jgx']
    # Make sure that the values of weights are floats
    for k, v in pars_conn_ext.items():
        pars_conn_ext[k] = float(v)

    # Produce N_conn_ext dictionary
    # NOTE: at present the routine only compute the number of neurons as one neuron per synapse
    N_conn_ext = {'ex': pars_extsyn['ex']['N_s'],
                  'ix': pars_extsyn['ix']['N_s'],
                  'gx': pars_extsyn['gx']['N_s']}

    return pars_extsyn, pars_prer_ext, pars_conn_ext, N_conn_ext, ext_conn, spk

def nean_parameters(N_e,N_i,N_g,connectivity=True,
                      N_clusters=0,
                      spatial_geometry=None,
                      ngn_setup = 'default',
                      topology={},p_conn={},w_scaling={},
                      ne_pars={},ni_pars='same',ng_pars={},glt_pars={},
                      ee_pars='default',ie_pars='same',ii_pars='default',ei_pars='same',
                      prer_ee_pars='default',prer_ie_pars='same',prer_ii_pars='default',prer_ei_pars='same',
                      randomize_ics={},
                      ex_topology='default',
                      external=False,
                      ex_pars={}, ix_pars='same', gx_pars={},
                      prer_ex_pars={}, prer_ix_pars='same', prer_gx_pars={},
                      **kwargs):
    """
    Generate parameters for NEA network model.

    :param N_e:
    :param N_i:
    :param N_g:
    :param connectivity:
    :param ngn_setup:
    :param topology:
    :param p_conn:
    :param w_scaling:
    :param ne_pars:
    :param ni_pars:
    :param ng_pars:
    :param glt_pars:
    :param ee_pars:
    :param ie_pars:
    :param ii_pars:
    :param ei_pars:
    :param prer_ee_pars:
    :param prer_ie_pars:
    :param prer_ii_pars:
    :param prer_ei_pars:
    :param randomize_ics:
    :param kwargs:
    :return:
    """

    # Network parameters
    plif,psyn,pglt,ppre,pconn,Nconn,edges,coords,geom = nea_parameters(N_e,N_i,N_g,connectivity=connectivity,
                                                           spatial_geometry=spatial_geometry,
                                                           ngn_setup = ngn_setup,
                                                           topology=topology,
                                                           p_conn=p_conn,w_scaling=w_scaling,
                                                           ne_pars=ne_pars,ni_pars=ni_pars,ng_pars=ng_pars,glt_pars=glt_pars,
                                                           ee_pars=ee_pars,ie_pars=ie_pars,ii_pars=ii_pars,ei_pars=ei_pars,
                                                           prer_ee_pars=prer_ee_pars,prer_ie_pars=prer_ie_pars,
                                                           prer_ii_pars=prer_ii_pars,prer_ei_pars=prer_ei_pars,
                                                           randomize_ics=randomize_ics,
                                                           **gu.merge_dicts(kwargs,{'N_clusters': N_clusters}))
    # External (stimulation) parameters
    # Handling of different model setups
    if ngn_setup in ['model-A', 'default']:
        ex_pars = gu.merge_dicts(ex_pars,{'N_s': pconn['C_e']})
        ix_pars = 'same'
        gx_pars = gu.merge_dicts(gx_pars,{'N_s': pconn['C_g']})
        # Generate external weights
        jxx = {'jex' : pconn['jee'],
               'jix' : pconn['jee'],
               'jgx': pconn['jgg']
               }
    psyn_x,ppre_x,pconn_x,Nconn_x,syn_edges_x,spk = extstim_parameters({'e': N_e, 'i': N_i, 'g': N_g},
                                                                        plif,N_clusters=geom['N_clusters'],
                                                                        external=external,
                                                                        ex_topology=ex_topology,
                                                                        ex_pars=ex_pars,ix_pars=ix_pars,gx_pars=gx_pars,
                                                                        prer_ex_pars=prer_ex_pars,prer_ix_pars=prer_ix_pars,prer_gx_pars=prer_gx_pars,
                                                                        **gu.merge_dicts(jxx,kwargs))

    # Create complete parameter dictionary to pass to spiking network simulator
    pars_mhv  = gu.merge_dicts(psyn,psyn_x)
    pars_prer = gu.merge_dicts(ppre,ppre_x)
    pars_conn = gu.merge_dicts(pconn,pconn_x)
    # syn_conn  = gu.merge_dicts(syn_edges,syn_edges_x)
    syn_conn = syn_edges_x
    Nconn     = gu.merge_dicts(Nconn,Nconn_x)

    for k,v in Nconn.items():
        Nconn[k] = np.uint32(v)

    # Compute threshold rate and mean / std quantities
    # The external connectivity needed to compute the threhsold rate and the final external stimulation rate as well as
    # mean and STD of external input is in the Nconn dictionary keys ['ex','ix','gx']
    for k in ['e','i','g']:
        if k != 'g':
            plif[k]['ratet'] = (plif[k]['vt']-plif[k]['vl'])/(Nconn[k+'x']*pconn['j'+k+'e']*plif[k]['taum'])
        else:
            # pconn['K_xx'] as provided by ngn_parameters is a vector...
            try:
                plif[k]['ratet'] = (plif[k]['vt']-plif[k]['vl'])/plif[k]['taum']/(np.sum(np.vstack([pconn['K_'+ks]*np.mean(pconn['w'+ks]) for ks in ['ee', 'ei', 'ii', 'ie']]),axis=0)+pconn['G']*pconn['jgg']).item() if Nconn[k] > 0 else 0.0 # must be a scalar!!
            except ValueError:
                # This is the case where you have clusters
                plif[k]['ratet'] = (plif[k]['vt'] - plif[k]['vl'])/plif[k]['taum']/(np.sum(np.vstack([pconn['K_' + ks]*np.mean(pconn['w' + ks]) for ks in ['ee', 'ei', 'ii', 'ie']]), axis=0) + pconn['G']*pconn['jgg']) if Nconn[k]>0 else np.zeros(N_clusters)  # In this case it is not a scalar
        # Generate correct entries for mx and sx for external inputs (note that they must be passed in units of 'v' -->
        # --> see equations in cell_node in network_classes_brian
        pars_mhv[k+'x']['rate'] = pars_mhv[k+'x']['rho']*plif[k]['ratet'] ## Convert to useful rate in external neuron modules when attached
        # TODO: WARNING: The next if statement reduces to scalar ix and sx for glia, when rho==0 : this is to save memory and increment speed in the clustered network
        if (k=='g')and(N_clusters>=2)and(np.sum(pars_mhv[k+'x']['rho'])==0.):
            try:
                plif[k]['ratet'] = plif[k]['ratet'].item(0)
            except AttributeError:
                # In case the above fails because ratet of 'g' population is already a scalar: handling of this parameter is different from 'e' and 'i'
                pass
            pars_mhv[k + 'x']['rho'] = 0.0
        plif[k]['ix'] = Nconn[k+'x']*pars_conn['j'+k+'x']*pars_mhv[k+'x']['rho']*plif[k]['ratet']*plif[k]['taum']
        plif[k]['sx'] = np.sqrt(pars_conn['j'+k+'x']*plif[k]['ix'])
        # Glia is conditionally handled in the class with deterministic module without ix and sx as parameters (an alternative could be to declare ix an sx also in the deterministic case)

    # # Make parameters suitable for assignment in the presence of clustering
    # plif = check_clustering(plif,geom,pars_to_check=['ix','sx'])

    # return plif,pars_mhv,pglt,pars_prer,pars_conn,Cconn,Nconn,edges,syn_conn,spk
    return plif,pars_mhv,pglt,pars_prer,pars_conn,Nconn,edges,syn_conn,spk,coords,geom

def nea_setup_parameters(N_e=0, N_i=0, N_g=0, nea_setup='default',
        connectivity=True,N_clusters=0,
        spatial_geometry=None,topology={},
        ne_pars={}, ni_pars={}, ng_pars={}, glt_pars={},
        ee_pars='default', ie_pars='same', ii_pars='default', ei_pars='same',
        prer_ee_pars='default', prer_ie_pars='same', prer_ii_pars='default', prer_ei_pars='same',
        randomize_ics={'e': True, 'i': True, 'g': True, 'gamma': False},
        p_conn={},
        ex_topology='default',
        ex_pars={}, ix_pars='same', gx_pars={},
        prer_ex_pars={}, prer_ix_pars='same', prer_gx_pars={},
        **kwargs):
    # Provides model parameters for different setups
    # It is a wrapper of different methods optimized to pass parameters to the Brian Class
    pars = {k: {} for k in ['cell','syn','ecs','conn','N','edges','syn_conn','coords','geom']}

    if nea_setup in ['l5e'] :
        pars['cell']['e'] = set_module_parameters(module_pars=ne_pars, module_default=neuron_parameters(model='l5e'),module_same=neuron_parameters(model='l5e'))
    elif nea_setup=='l5e+ecs':
        pars['cell'],pars['syn'],pars['ecs'],_,pars['N'],_,_,_ = nea_parameters(1,0,0,nea_setup=nea_setup,ne_pars=ne_pars,
                                                                        spatial_geometry=None,connectivity=False)
        # TODO: Automatic assignment of connections in special scenario (for automatic later handling of monitors)
        pars['N']['ee'] = 1
        pars['cell']['e']['HBC0_e'] = pars['ecs']['HBC0_e']
        # TODO: Revise according to different synaptic schemes (once l5i is included)
        try:
            pars['syn']['ee']['D'] = pars['ecs']['D_Glu_e']
        except:
            pass
        try:
            pars['syn']['ei']['D'] = pars['ecs']['D_GABA_e']
        except:
            pass
        # Clean unused entries
        for k in ['i','g']:
            del pars['cell'][k]
        for k in ['ei','ii','ie']:
            del pars['syn'][k]

    elif nea_setup=='l5e+i':
        pass
    elif nea_setup=='single-nea':

        '''
        This configuration only needs to specify:
        - N_e
        - ne_pars
        - ng_pars
        - ex_pars
        - glt_pars
        - prer_ex_pars
        - kwargs as wex,wgex,wexg

        WARNING: It will NOT provide any connectivity.
        '''

        # Pre-processing
        # Assures that if N_g and N_e are larger than zero both, than they must be equal
        if N_g*N_e>0: N_g = N_e

        pars['lif'] = {}
        pars['lif']['e'] = neu.lif_parameters(ICs=0.0, NS=0, taus=0.0, **ne_pars)
        # Default Astrocyte parameters
        pars['lif']['g'] = neu.lif_parameters(ICs=0.0, NS=0, taus=0.0, **ng_pars)

        # Initialize synaptic parameters (to neurons)
        pars['syn'] = {}
        pars['syn']['nx'] = gu.merge_dicts(syn.mhv_parameters(**ex_pars),extsyn_parameters(**ex_pars))
        # Initialize synaptic parameters to glia
        pars['syn']['gx'] = gu.merge_dicts(syn.mhv_parameters(**gu.merge_dicts({'u0': 1.0},gx_pars)),extsyn_parameters(**gu.merge_dicts({'N_s': 0},gx_pars)))

        # Initialize gliotransmitter pathway
        pars['glt'] = gliot_parameters(**glt_pars)

        # Initialize presynaptic weights
        pars['prer'] = {}
        pars['prer']['nx'] = prer_parameters(**prer_ex_pars)

        # Initialize connectivity parameters
        pars['conn'] = {'jx': (pars['lif']['e']['vt']-pars['lif']['e']['vr'])/np.sqrt(pars['syn']['nx']['N_s']), # X --> N(E)
                        'wx': (pars['lif']['g']['vt']-pars['lif']['g']['vr'])/pars['syn']['gx']['N_s'],          # X --> G
                        'wgx': (pars['lif']['g']['vt']-pars['lif']['g']['vr'])/pars['syn']['nx']['N_s'],         # S --> G
                        'wxg': pars['glt'].pop('js')}                                                            # G --> S
        pars['conn'] = gu.varargin(pars['conn'], **kwargs)

        # Initilize Setup Number
        pars['N'] = {}
        pars['N']['e'] = int(N_e)
        pars['N']['g'] = int(N_g)
        # # External inputs (used for plotting purposes?)
        # pars['N']['nx'] = int(pars['N']['e']*pars['syn']['nx']['N_s'])
        pars['N']['gx'] = int(pars['N']['g']*pars['syn']['gx']['N_s'])
        # External inputs WITHOUT glia regulation
        # pars['N']['nx'] = np.floor((1-pars['syn']['nx']['syn_frac'])*pars['N']['e']*pars['syn']['nx']['N_s']).astype(int)
        pars['N']['nx'] = np.round((1 - pars['syn']['nx']['syn_frac'])*pars['N']['e']*pars['syn']['nx']['N_s'],decimals=0).astype(int)
        # External inputs WITH glia regulation
        pars['N']['sx'] = int(pars['N']['e']*pars['syn']['nx']['N_s'] - pars['N']['nx'])

        pars['edges'], pars['syn_conn'] = {}, {}
        # From external neurons / cells to neurons/glia
        nnx = np.arange(pars['N']['nx']).astype(np.int32)
        nsx = np.arange(pars['N']['sx']).astype(np.int32)
        ngx = np.arange(pars['N']['gx']).astype(np.int32)
        # N_n = np.floor((1-pars['syn']['nx']['syn_frac'])*pars['syn']['nx']['N_s']).astype(int)
        N_n = np.round((1 - pars['syn']['nx']['syn_frac'])*pars['syn']['nx']['N_s'],decimals=0).astype(int)
        N_s = pars['syn']['nx']['N_s'] - N_n
        if connectivity and pars['N']['nx']>0:
            pars['edges']['nx'] = np.vstack((nnx,np.floor_divide(nnx,N_n).astype(np.int32)))
        else:
            pars['edges']['nx'] = np.zeros((2,0),dtype=np.int32)
        if connectivity and pars['N']['gx']>0:
            pars['edges']['gx'] = np.vstack((ngx,np.floor_divide(ngx,int(pars['syn']['gx']['N_s'])).astype(np.int32)))
        else:
            pars['edges']['gx'] = np.zeros((2,0),dtype=np.int32)
        # Shared synapses
        pars['edges']['sx'] = np.zeros((2,0),dtype=np.int32)
        pars['edges']['gs'] = np.zeros((2,0),dtype=np.int32)
        pars['edges']['sg'] = np.zeros((2, 0), dtype=np.int32)
        if connectivity and pars['N']['sx']>0:
            # From X to N
            pars['edges']['sx'] = np.vstack((nsx,np.floor_divide(nsx,N_s).astype(np.int32)))
            # Same synapses but indexed from X to G (this assume that N_e and N_g are identical, so neurons and glial cells are identically indexed)
            if pars['conn']['wgx']>0.:
                pars['edges']['gs'] = np.vstack((nsx, np.floor_divide(nsx, N_s).astype(np.int32)))
            if pars['conn']['wxg']>0.:
                pars['edges']['sg'] = np.vstack((np.floor_divide(nsx, N_s).astype(np.int32),nsx))

        # # Build connectivity
        # pars['edges'], pars['syn_conn'] = {}, {}
        # # From external neurons / cells to neurons/glia
        # nex = np.arange(pars['N']['ex']).astype(np.int32)
        # ngx = np.arange(pars['N']['gx']).astype(np.int32)
        # pars['edges']['ex'] = np.vstack((nex,np.floor_divide(nex,int(pars['syn']['ex']['N_s'])).astype(np.int32)))
        # pars['edges']['gx'] = np.vstack((ngx,np.floor_divide(ngx,int(pars['syn']['gx']['N_s'])).astype(np.int32)))
        # # From synapses to glia
        # if pars['syn']['ex']['syn_frac']>0.:
        #     pars['edges']['gex'] = (pars['edges']['ex'].T[np.where(np.mod(nex,pars['syn']['ex']['N_s'])<np.floor(pars['syn']['ex']['syn_frac']*pars['syn']['ex']['N_s']))[0]]).T
        # else:
        #     pars['edges']['gex'] = np.zeros((2,0),dtype=np.int32)
        # if (pars['syn']['ex']['syn_frac']>0.)and(pars['conn']['wexg']>0.):
        #     pars['edges']['exg'] = np.flipud(pars['edges']['gex'])
        # else:
        #     pars['edges']['exg'] = np.zeros((2,0),dtype=np.int32)

        # Assign dummy geometry (for compatibility in class methods)
        # 'geometry' is currently not used in the single-astro configuration
        pars['geom'] = geometry_parameters(N_e,0,N_g,None)

    elif nea_setup in ['model-A', 'default']:
        '''
        Model A by Brunel, JCN 2000. This also assumes
        Only p_conn = {'ee' : prob} needs be specified.
        The input current must be computed from the rate separately, and provided as 'ix' and 'sx' input to ne_pars and ni_pars
        '''

        pars['lif'], pars['syn'], pars['glt'], \
        pars['prer'], pars['conn'], \
        pars['N'], pars['edges'], pars['syn_conn'], _,\
        pars['coords'], pars['geom'] = nea_setup_parameters(N_e, N_i, N_g,
                                                        p_conn=p_conn,
                                                        connectivity=connectivity,
                                                        N_clusters=N_clusters,
                                                        spatial_geometry=spatial_geometry,
                                                        topology=topology,
                                                        ngn_setup='model-A',
                                                        ne_pars=ne_pars, ni_pars=ni_pars,
                                                        ng_pars=ng_pars, glt_pars=glt_pars,
                                                        ee_pars=ee_pars, ie_pars='same',
                                                        ii_pars=ii_pars, ei_pars='same',
                                                        prer_ee_pars=prer_ee_pars, prer_ie_pars='same',
                                                        randomize_ics=randomize_ics,
                                                        external=False, ex_topology=ex_topology,
                                                        ex_pars=ex_pars,
                                                        ix_pars='same',
                                                        gx_pars=gx_pars,
                                                        **kwargs)

        # Temporary work around to convert to native python data types
        for k,p in pars.items():
            if isinstance(p,dict):
                for kk,pp in p.items():
                    if isinstance(pp,np.generic): pp = pp.item()
            else:
                if isinstance(pp, np.generic): p = p.item()

    return pars

if __name__=="__main__":
    # #--------------------------------------------------------------------------------------------
    # # Testing different E_GAT
    # #--------------------------------------------------------------------------------------------
    # p_ecs = ecs_parameters()
    # p_ast = astrocyte_parameters()
    #
    # E_Na = NernstPotential(p_ecs['N0_e'],p_ast['N0_a'],1,p_ast['T_exp'])
    # E_Cl = NernstPotential(p_ecs['C0_e'], p_ast['C0_a'], -1, p_ast['T_exp'])
    # E_GAT = 0.5*(3*E_Na+E_Cl+ThermalPotential(p_ast['T_exp'])*np.log(p_ecs['GABA0_e']/p_ast['GABA0_a']))
    # print('E_GAT\t',E_GAT)

    #--------------------------------------------------------------------------------------------
    # Testing of nea
    #--------------------------------------------------------------------------------------------
    # pars_cell, pars_syn, pars_ecs, pars_conn, N_conn, edges, coords, network_geometry = nea_parameters(1,0,0,connectivity=False,
    #     nea_setup='l5e+ecs',
    #     spatial_geometry=None)
    pars = nea_setup_parameters(1,0,0,nea_setup='l5e+ecs')
    print(pars)
    # print('check')