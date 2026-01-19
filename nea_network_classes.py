import numpy as np
import scipy.constants as spc
import copy as cp
import multiprocessing as mpr

from brian2 import *
import nea_parameters as neapars
import nea_classes as neamod

# Custom modules
import os,sys
sys.path.append(os.path.join(os.path.expanduser('~'),'Ongoing.Projects/pycustommodules'))
import save_utils as svu
import general_utils as gu
import brian_utils as bu

## Warning handling
import warnings as wrn
wrn.filterwarnings("ignore")
BrianLogger.suppress_name('resolution_conflict')

#-----------------------------------------------------------------------------------------------------------------------
# Testing-related modules (temporary)
#-----------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------------------------------------
safe_threads = lambda n,safe_cpu : int(min(max(1,n),mpr.cpu_count()-1)) if safe_cpu else int(min(max(1,n),mpr.cpu_count()))  ## This will assure that the number of threads is 1<=n<=n_cpu-1
# -----------------------------------------------------------------------------------------------------------------------
# NEA Network Simulator Main Class
# -----------------------------------------------------------------------------------------------------------------------
class NEAGroup(object):
    def __init__(self,
                 N_e=0,N_i=0,N_g=0,nea_setup='default',
                 ne_pars={},ni_pars={},ng_pars={},ecs_pars={},
                 ee_pars='default',ie_pars='same',ii_pars='default',ei_pars='same',
                 prer_ee_pars='default',prer_ie_pars='same',prer_ii_pars='default',prer_ei_pars='same',
                 randomize_ics={'e': True,'i': True,'g': True, 'gamma': False},
                 p_conn={},
                 ex_pars={},ix_pars='same',gx_pars={},
                 prer_ex_pars={},prer_ix_pars='same',prer_gx_pars={},
                 network_connectivity=True,external_topology='default',
                 network_geometry=None,network_topology={},N_clusters=0,
                 dynamic_reset={'e': False, 'i': False, 'g': False},
                 glia_block=False,
                 code_dir='./codegen',
                 pars_dict=None,
                 eqs_noise={'e': False, 'i': False, 'g': False},
                 **kwargs):

            # Initialize brian2 environment
            set_device('cpp_standalone', directory=code_dir, build_on_run=False)
            device.delete(force=True) # Clean codegen directory
            self._code_dir = code_dir
            self.__build = False

            # Save configuration
            self._nea_setup = nea_setup
            self.__glia_block = glia_block
            # Noise settings in the equations (currently used only in the EI(G) network)
            self.__eqs_noise = gu.varargin({'e': False, 'i': False, 'g': False}, **eqs_noise)
            # Generate parameter dictionary
            self.pars = {}

            if any(l in nea_setup for l in ['l5e','l5i']):
                #-------------------------------------------------------------------------------------------------------
                # Single Neuron (excitatory 'e' or inhibitory 'i')
                #-------------------------------------------------------------------------------------------------------
                if 'e' in nea_setup:
                    N_e = 1
                    N_i = 0
                elif 'i' in nea_setup:
                    N_e = 0
                    N_i = 1
                if pars_dict==None:
                    self.pars = neapars.nea_setup_parameters(N_e,N_i,0,nea_setup=nea_setup,
                                                     connectivity=network_connectivity,N_clusters=0,
                                                     ne_pars = ne_pars,ng_pars=ng_pars, ecs_pars=ecs_pars,
                                                     ex_pars=ex_pars,gx_pars=gx_pars,prer_ex_pars=prer_ex_pars,
                                                     prer_gx_pars=prer_gx_pars,
                                                     **kwargs)
                else:
                    # Load parameters from MF curves
                    # TODO: Up to ecs_pars assignments are correct. After they need to be revised based on the final structure of pars_dict
                    self.pars = neapars.ngn_setup_parameters(N_e,N_i,0,ngn_setup=nea_setup,
                                                     connectivity=network_connectivity,N_clusters=0,
                                                     ne_pars=pars_dict['cell']['e'],ni_pars=pars_dict['cell']['i'],ng_pars=pars_dict['cell']['g'],
                                                     ecs_pars=pars_dict['ecs'],
                                                     ex_pars=pars_dict['syn']['nx'],gx_pars=pars_dict['syn']['gx'],
                                                     prer_ex_pars=pars_dict['prer']['nx'],
                                                     prer_gx_pars={},
                                                     **pars_dict['conn'])
                    # TODO: Connectivity redefinition
                # # We keep the matrices of connections aside for faster handling of parameters
                # if network_connectivity:
                #     self._syn_to_glia = ngp.retrieve_syn_glia_connections(self.pars['edges'], pathway='syn-to-glia', setup='sad')
                #     self._glia_to_syn = ngp.retrieve_syn_glia_connections(self.pars['edges'], pathway='glia-to-syn', setup='sad')

            elif 'model-A' in nea_setup:
                pass
                # # TODO
                # if nea_setup=='model-A':
                #     if pars_dict==None:
                #         self.pars = neapars.ngn_setup_parameters(N_e=N_e,N_i=N_i,N_g=N_g,ngn_setup='model-A',
                #                                              p_conn=p_conn,connectvity=network_connectivity,N_clusters=N_clusters,
                #                                              spatial_geometry=network_geometry,topology=network_topology,
                #                                              ne_pars=ne_pars,ni_pars=ni_pars,ng_pars=ng_pars,
                #                                              ecs_pars=ecs_pars,
                #                                              ee_pars=ee_pars,ii_pars=ii_pars,
                #                                              prer_ee_pars=prer_ee_pars,prer_ie_pars=prer_ie_pars,
                #                                              prer_ii_pars=prer_ii_pars,prer_ei_pars=prer_ei_pars,
                #                                              randomize_ics=randomize_ics,
                #                                              ex_topology=external_topology,
                #                                              ex_pars=ex_pars,
                #                                              gx_pars=gx_pars,
                #                                              **kwargs)
                #     else:
                #         raise Exception("Conversion from MF to Spiking model parameters NOT implemented for the NGN network")
                #     # We keep the matrices of connections aside for faster handling of parameters
                #     if network_connectivity:
                #         self._syn_to_glia = ngp.retrieve_syn_glia_connections(self.pars['edges'], pathway='syn-to-glia',setup='ngn')
                #         self._glia_to_syn = ngp.retrieve_syn_glia_connections(self.pars['edges'], pathway='glia-to-syn',setup='ngn')

            # Work on a deep copy of parameters
            # Give dimensions to relevant parameters
            # pars = addParameterUnits(cp.deepcopy(self.pars),ngn_setup=ngn_setup)
            pars = cp.deepcopy(self.pars)

            # Build effective Network
            if network_connectivity:
                # TODO: Set dynamic reset
                self._dynamic_reset = gu.varargin({'e': False, 'i': False, 'g': False}, **dynamic_reset)
                self.network = self._buildNetwork(nea_setup,pars)
            else:
                print("WARNING: Network NOT built because network_connectivitity==False")

            # Save a copy of original parameters (with NO units)
            self._original_pars = cp.deepcopy(self.pars)

    def _synaptic_namespace(self,pars,syn_name):
        # TODO: MARKED FOR DELETION
        """
        Provide correct namespace for synaptic group depending on self.__glia_block value

        Inputs:
        - syn_name : 'ee' | 'ie' | 'ei' | 'ii'
        - pars     : dictionary of model parameters WITH UNITS

        Return:
        - par_dict : dictionary of parameters to pass to the 'namespace' input argument of the synaptic group

        Maurizio De Pitta', Basque Center of Applied Mathematics, 02.02.2020
        """
        return gu.merge_dicts(pars['prer'][syn_name],{'u0': pars['syn'][syn_name]['u0']})

    def _gliot_namespace(self,pars,syn_name):
        # TODO: MARKED FOR DELETION
        return gu.merge_dicts(pars['prer'][syn_name],pars['glt'])

    def _buildNetwork(self,nea_setup,pars):
        """
        This is the effective module that builds the brian2-network object. It is meant to be internally called by the class. It is divided from the __init__ module
        to allow rebuilds of the network, without the need to specify all the parameters and settings.

        :param ngn_setup:
        :param pars:
        :return:
        """
        # Setup Network
        network_objects = []

        if nea_setup=='l5e':
            pass
        elif nea_setup=='l5i':
            pass
        elif nea_setup=='l5e+ecs':
            # ----------------------------------------------------------------------------------------------------------
            # Setup compartments
            # ----------------------------------------------------------------------------------------------------------
            # Generate neuronal compartments
            NX = neamod.periodic_nodes(1,10*Hz,name='src*',dt=1*ms)
            # Single neuron coupling
            N = neamod.neuron_cell(1,pars['cell']['e'],model='l5e',name='neu*',method='euler')
            N.v = -70*mV

            # Generate ecs compartments
            ECS = neamod.ecs_space(1,pars['ecs'],coupling='n',name='ecs*',method='euler')
            # # Generate a distal ecs compartment
            ECSO = neamod.ecs_space(1,pars['ecs'],coupling='none',name='ecso*',method='euler')

            # Add objects
            network_objects.extend([NX,N,ECS,ECSO])

            # ----------------------------------------------------------------------------------------------------------
            # Setup connections
            # ----------------------------------------------------------------------------------------------------------
            # Connect neuron with ecs
            neamod.cell_ecs_connection(N,ECS,source_id='n')

            # Connect ECS compartments
            E2E = neamod.ecs_ecs_connection(ECS,ECSO,pars['ecs'],name='e2e*')
            E2E.connect(i=[0],j=[0])

            # Synapses
            S = neamod.synaptic_connection(NX,ECS,pars['syn']['ee'],sinput='glu',name='syn*')
            S.connect(i=[0],j=[0])
            # Assign default parameters
            # TODO: correct dictionary entries once finalized in the parameter module
            S.g = pars['syn']['ee']['g']  ## NOTE: This g also reflect the g in g_GABA for the astrocyte!!!
            S.Nt_rel = pars['syn']['ee']['Nt_rel']
            S.Lambda_s = pars['syn']['ee']['Lambda_s']
            # TODO: ttype below won't work -- delete this comment once fixed
            S.Nt0_e = pars['ecs']['G0_e'] if ttype=='glu' else pars['ecs']['GABA0_e']
            # Set initial conditions
            S.r = 0.0
            S.nt_s = 0.0*mole

            # Add objects
            network_objects.extend([E2E,S])

        # if ngn_setup=='single-astro':
        #     if (hasattr(pars['syn']['nx']['rate'],'__len__'))and(np.size(pars['syn']['nx']['rate'])>1):
        #         if (pars['syn']['nx']['T_rate']!=None):
        #             # Stimulation on Neurons
        #             _stimulus_nx = TimedArray(pars['syn']['nx']['rate']*hertz, dt=pars['syn']['nx']['T_rate']*second)
        #             rates_nx = _stimulus_nx
        #             rates_sx = _stimulus_nx
        #         elif (np.size(pars['syn']['nx']['rate'])==pars['N']['e'])and(pars['syn']['nx']['T_rate']==None):
        #             rates_nx = np.tile(pars['syn']['nx']['rate'],(pars['N']['nx']//pars['N']['e'],1)).ravel(order='F')*hertz
        #             rates_sx = np.tile(pars['syn']['nx']['rate'],(pars['N']['sx']//pars['N']['e'],1)).ravel(order='F')*hertz
        #     else:
        #         # Rate is a scalar (the line below is made to ensure robustness)
        #         xrate = np.atleast_1d(pars['syn']['nx']['rate'])
        #         rates_nx = xrate[-1]*hertz
        #         rates_sx = xrate[-1]*hertz
        #
        #     # Stimulation on Glia
        #     if (self.__glia_block==False)and(self._glia_to_syn):
        #         # Currently assume the same stimulation on
        #         if (hasattr(pars['syn']['gx']['rate'],'__len__'))and(np.size(pars['syn']['gx']['rate'])>1):
        #             if (pars['syn']['gx']['T_rate']!=None):
        #                 # In this case the '_stimulus_gx(t)' is created in the 'self.simulate()' method
        #                 _stimulus_gx = TimedArray(pars['syn']['gx']['rate']*hertz, dt=pars['syn']['gx']['T_rate']*second)
        #                 rates_gx = _stimulus_gx
        #             elif (np.size(pars['syn']['gx']['rate'])==pars['N']['g'])and(pars['syn']['gx']['T_rate']==None):
        #                 rates_gx = np.tile(pars['syn']['gx']['rate'],(pars['N']['gx']//pars['N']['g'],1)).ravel(order='F')*hertz
        #         else:
        #             # Rate is a scalar
        #             gxrate = np.atleast_1d(pars['syn']['gx']['rate'])
        #             rates_gx = gxrate[-1]*hertz
        #
        #     # Build modules
        #     # Postsynaptic neurons
        #     E = cell_nodes(pars['N']['e'], pars['lif']['e'], name='E',method='exact',dynamic_reset=self._dynamic_reset['e'])
        #     network_objects.extend([E])
        #
        #     # External Neurons
        #     if pars['N']['nx']>0:
        #         NX = poisson_nodes(pars['N']['nx'],rates_nx,name='NX')
        #         S_nx = probabilistic_synapses(NX, E, name='Sn')
        #         S_nx.connect(i=pars['edges']['nx'][0], j=pars['edges']['nx'][1])
        #         S_nx.u_0 = pars['syn']['nx']['u0']
        #         S_nx.w = pars['conn']['jx']
        #
        #         # Add modules
        #         network_objects.extend([NX,S_nx])
        #
        #     # Glia cell
        #     if pars['N']['g']>0:
        #         G = cell_nodes(pars['N']['g'], pars['lif']['g'], name='G',dynamic_reset=self._dynamic_reset['g'])
        #         network_objects.append(G)
        #         if pars['N']['gx'] > 0:
        #             GX = poisson_nodes(pars['N']['gx'],rates_gx, name='GX')
        #             S_gx = probabilistic_synapses(GX, G, name='Sg')
        #             S_gx.connect(i=pars['edges']['gx'][0], j=pars['edges']['gx'][1])
        #             S_gx.u_0 = pars['syn']['gx']['u0']
        #             S_gx.w = pars['conn']['wx']
        #             # Add modules
        #             network_objects.extend([GX,S_gx])
        #     else:
        #         G = None # This is needed to have the TripartiteConnection(...) below otherwise G is not found
        #
        #     # Add shared connections
        #     if pars['N']['sx'] > 0:
        #         SX = poisson_nodes(pars['N']['sx'],rates_sx,name='SX')
        #         # S_sx is a list of variable objects
        #         S_sx = TripartiteConnection(SX,E,G,pars['N']['sx'],gu.merge_dicts(pars['syn']['nx'],pars['prer']['nx']),
        #                                     pars['edges']['sx'][0],pars['edges']['sx'][1],
        #                                     pars['edges']['sg'][0],pars['edges']['gs'][1],
        #                                     pars['edges']['sg'][1],pars['edges']['gs'][0],
        #                                     jxx=pars['conn']['jx'],wxx=pars['conn']['wgx'],wsg=pars['conn']['wxg'],
        #                                     shared=(self.__glia_block==False)and(self._syn_to_glia)and(pars['conn']['wgx']>0.),
        #                                     gliot=(self.__glia_block==False)and(self._glia_to_syn)and(pars['conn']['wxg']>0.),
        #                                     gliot_pars=gu.merge_dicts(pars['prer']['nx'],pars['glt']),
        #                                     name='Ss')
        #         network_objects.append(SX)
        #         network_objects.extend(S_sx)
        #
        # elif 'model-A' in ngn_setup:
        #     if ngn_setup=='model-A':
        #         # Neurons
        #         E = cell_nodes(pars['N']['e'],pars['lif']['e'],name='E',noise=self.__eqs_noise['e'],dynamic_reset=self._dynamic_reset['e'])
        #         I = cell_nodes(pars['N']['i'],pars['lif']['i'],name='I',noise=self.__eqs_noise['i'],dynamic_reset=self._dynamic_reset['i'])
        #         E.v = pars['lif']['e']['ICs']
        #         I.v = pars['lif']['i']['ICs']
        #         # Noise assignment
        #         E.ix = pars['lif']['e']['ix']
        #         I.ix = pars['lif']['i']['ix']
        #         E.sx = pars['lif']['e']['sx']
        #         I.sx = pars['lif']['i']['sx']
        #         # # Append to network objects
        #         network_objects.extend([E,I])
        #         if (pars['N']['g']>0)and(self.__glia_block==False):
        #             if (np.sum(pars['lif']['g']['sx'])==0.):
        #                 G = cell_nodes(pars['N']['g'],pars['lif']['g'],name='G',noise=self.__eqs_noise['g'],dynamic_reset=self._dynamic_reset['g'])
        #                 G.ix = pars['lif']['g']['ix']
        #             else:
        #                 G = cell_nodes(pars['N']['g'], pars['lif']['g'], name='G', noise=True,dynamic_reset=self._dynamic_reset['g'])
        #                 # Noise assignment
        #                 G.ix = pars['lif']['g']['ix']
        #                 G.sx = pars['lif']['g']['sx']
        #             G.v = pars['lif']['g']['ICs']
        #             # Append to network objects
        #             network_objects.append(G)
        #         else:
        #             G = None
        #
        #         # Synaptic connections
        #         # EE
        #         S_ee = TripartiteConnection(E,E,G,pars['N']['ee'],self._synaptic_namespace(pars,'ee'),
        #                 pars['edges']['ee'][0],pars['edges']['ee'][1],
        #                 pars['edges']['eeg'][0],pars['edges']['gee'][1],
        #                 pars['edges']['eeg'][1],pars['edges']['gee'][0],
        #                 jxx=pars['conn']['jee'], wxx=pars['conn']['wee'], wsg=pars['conn']['wsg'],
        #                 shared=(pars['N']['g']>0)and(self._syn_to_glia['ee'])and(not self.__glia_block),
        #                 gliot=(pars['N']['g']>0)and(self._glia_to_syn['ee'])and(not self.__glia_block),
        #                 gliot_pars=self._gliot_namespace(pars,'ee'),
        #                 name='ee')
        #         network_objects.extend(S_ee)
        #
        #         # IE
        #         S_ie = TripartiteConnection(E, I, G, pars['N']['ie'], self._synaptic_namespace(pars, 'ie'),
        #             pars['edges']['ie'][0], pars['edges']['ie'][1],
        #             pars['edges']['ieg'][0], pars['edges']['gie'][1],
        #             pars['edges']['ieg'][1], pars['edges']['gie'][0],
        #             jxx=pars['conn']['jie'], wxx=pars['conn']['wie'], wsg=pars['conn']['wsg'],
        #             shared=(pars['N']['g'] > 0) and (self._syn_to_glia['ie']) and (not self.__glia_block),
        #             gliot=(pars['N']['g'] > 0) and (self._glia_to_syn['ie']) and (not self.__glia_block),
        #             gliot_pars=self._gliot_namespace(pars,'ie'),
        #             name='ie')
        #         network_objects.extend(S_ie)
        #
        #         # EI
        #         S_ei = TripartiteConnection(I, E, G, pars['N']['ei'], self._synaptic_namespace(pars, 'ei'),
        #             pars['edges']['ei'][0], pars['edges']['ei'][1],
        #             pars['edges']['eig'][0], pars['edges']['gei'][1],
        #             pars['edges']['eig'][1], pars['edges']['gei'][0],
        #             jxx=-pars['conn']['jei'], wxx=pars['conn']['wei'], wsg=pars['conn']['wsg'],
        #             shared=(pars['N']['g'] > 0) and (self._syn_to_glia['ei']) and (not self.__glia_block),
        #             gliot=(pars['N']['g'] > 0) and (self._glia_to_syn['ei']) and (not self.__glia_block),
        #             gliot_pars=self._gliot_namespace(pars,'ei'),
        #             name='ei')
        #         network_objects.extend(S_ei)
        #
        #         # II
        #         S_ii = TripartiteConnection(I, I, G, pars['N']['ii'], self._synaptic_namespace(pars, 'ii'),
        #             pars['edges']['ii'][0], pars['edges']['ii'][1],
        #             pars['edges']['iig'][0], pars['edges']['gii'][1],
        #             pars['edges']['iig'][1], pars['edges']['gii'][0],
        #             jxx=-pars['conn']['jii'], wxx=pars['conn']['wii'], wsg=pars['conn']['wsg'],
        #             shared=(pars['N']['g'] > 0) and (self._syn_to_glia['ii']) and (not self.__glia_block),
        #             gliot=(pars['N']['g'] > 0) and (self._glia_to_syn['ii']) and (not self.__glia_block),
        #             gliot_pars=self._gliot_namespace(pars,'ii'),
        #             name='ii')
        #         network_objects.extend(S_ii)
        #
        #     # Generate gap junction connections between glial cells
        #     if (pars['N']['g']>0)and(self.__glia_block==False):
        #         if (np.size(pars['edges']['gg'])>0)and(np.any(pars['edges']['gg'])>0):
        #             GJ_gg = gap_junctions(G, G, w=pars['conn']['jgg'], name='gjc')
        #             GJ_gg.connect(i=pars['edges']['gg'][0],j=pars['edges']['gg'][1])
        #             network_objects.append(GJ_gg)

        # Setup network object
        return Network(network_objects)

    # def updateGliaBlock(self,glia_block):
    #     """
    #     If given glia_block is different from the one internally stored in self.__glia_block, rebuild the network model with the same parameters
    #     but with the new glia_block configuration.
    #
    #     Inputs:
    #     - glia_block : True | False
    #
    #     Return:
    #     - Does not return anything. Simply modifies in case self.network object (issuing a warning)
    #     """
    #     if glia_block!=self.__glia_block:
    #         print("WARNING: glia_block has changed and network is being rebuilt")
    #         # Update __glia_block
    #         self.__glia_block = glia_block
    #         # Delete previous Network object
    #         delattr(self,'network')
    #         # Delete created code
    #         device.delete()
    #         # Delete created code and Restart device for new build
    #         device.reinit()
    #         device.activate(directory=self._code_dir, build_on_run=False)
    #         self.__build = False
    #         # Build effective Network
    #         self.network = self._buildNetwork(self._ngn_setup, addParameterUnits(cp.deepcopy(self.pars), ngn_setup=self._ngn_setup))

    # TODO: Update Noise?
    # def updateExternalNoise(self,rho,cell='e',update=None):
    #     # Calculate mean input current value (given rho) based on latest model parameters
    #     assert (cell in ['e','i','g']), "Stimulation cell must be 'e', 'i' or 'g'"
    #     # Work on a vectorialized version of 'update': If update is None it will create a dummy update=['full'] array. This is needed for
    #     # running loops in case of a clustered network
    #
    #     # Pre-processing of rho -- turn it into a vector for later looping
    #     rho_ = np.atleast_1d(rho)   # Vectorized version of rho
    #     # Pre-processing of update: None is equivalent to 'full' -->
    #     update = [1.] if update is None else np.atleast_1d(update)
    #     # --> 'full' updates are equivalent to fraction of neuron==1. Update is a numerical vector
    #     update_ = np.atleast_1d([1.0 if updt=='full' else updt for updt in update])
    #     if np.size(update)!=np.size(rho_):
    #         print('WARNING: update and rho_ do not have same dimensions: using the first update for all rhos')
    #         update_ = np.atleast_1d([update_.item(0)]*np.size(rho_))    # Vectorized version of update
    #     # Generate mean and average values by cluster
    #     ix_ = self.pars['N'][cell + 'x']*self.pars['conn']['j' + cell + 'x']*rho_*self.pars['lif'][cell]['ratet']*self.pars['lif'][cell]['taum']
    #     sx_ = np.sqrt(self.pars['conn']['j' + cell + 'x']*ix_)
    #     # The following take care of turning values
    #     if self.pars['geom']['N_clusters']>0:
    #         # Full mean/noise vectors (Initialization)
    #         # Retrieve current ix and sx values
    #         a_ = np.zeros(self.pars['N'][cell])
    #         b_ = np.zeros(self.pars['N'][cell])
    #         # a_ = self.network[cell.upper()].ix # Mean
    #         # b_ = self.network[cell.upper()].sx # Noise
    #         for i,cluster_size in enumerate(self.pars['geom']['N_'+cell+'_cluster']):
    #             if (cluster_size>0)and(update_[i]>0.):
    #                 # # Random selection (disabled)
    #                 # idx = numpy.random.choice(self.pars['geom'][cell+'_clusters'][i], size=int(update[i]*cluster_size), replace=False)
    #                 # Pick the first update elements of the cluster
    #                 idx = self.pars['geom'][cell+'_cluster'][i][:int(update[i]*cluster_size)]
    #                 a_[idx] = ix_[i]
    #                 b_[idx] = sx_[i]
    #         # Update ix and sx in the class -- this case already provides ix and sx of size N_cells
    #         self.pars['lif'][cell]['ix'] = cp.copy(a_)
    #         self.pars['lif'][cell]['sx'] = cp.copy(b_)
    #     else:
    #         # This case rho_ is just a scalar array and so is update
    #         if update_.item()>0:
    #             self.pars['syn'][cell+'x']['rho'] = rho_.item()
    #             self.pars['syn'][cell+'x']['rate'] = rho_.item()*self.pars['lif'][cell]['ratet']
    #             self.pars['lif'][cell]['ix'] = ix_.item()
    #             self.pars['lif'][cell]['sx'] = sx_.item()

    # # TODO: Update Parameters
    # def updateParameters(self, par_name, par_value, update=None):
    #     """
    #
    #     :param par_name:
    #     :param par_value:
    #     update   : Special argument to pass special update handlings (only for par_name=='rho)
    #     :return:
    #     """
    #     # Update model parameters through the namespace of individual groups in the network class
    #     # This is used in analysis of bifurcation diagrams
    #     # WARNING: the original copy of the parameters is in self.pars and contains dimensionless parameters
    #     if self._ngn_setup=='model-A':
    #         if par_name=='gj':
    #             for k in ['ei','ii']:
    #                 self.pars['conn']['j'+k] = self.pars['conn']['j'+k]/self.pars['conn'][par_name]*par_value
    #                 if (not self.__glia_block)and((self._syn_to_glia[k])or(self._glia_to_syn[k])):
    #                     self.network[k+'_n'].w = -self.pars['conn']['j'+k]  # to_post connections 'ei_n', 'ii_n'
    #                 else:
    #                     self.network[k].w = -self.pars['conn']['j' + k]     # standard connections 'ei','ii' in the EI-only case
    #             self.pars['conn'][par_name] = par_value
    #         if par_name=='rho':
    #             for k in ['e','i']:
    #                 self.updateExternalNoise(par_value,cell=k,update=update)
    #                 self.network[k.upper()].ix = self.pars['lif'][k]['ix']
    #                 self.network[k.upper()].sx = self.pars['lif'][k]['sx']
    #         if 'u0_' in par_name:
    #             # Update of the parameter u0 assumes that you pass it as 'u0_ee', 'u0_ie' etc.
    #             k = par_name.split('_')[-1]
    #             self.network[k].namespace['u0'] = par_value
    #             if not self._glia_to_syn[k]:
    #                 self.network[k].u_0 = par_value
    #                 # The case with gliotransmission will be automatically updated for the way u_0 is defined in this case

    def _updatedInitialConditions(self,ICs_dict):
        """
        Update network with given initial conditions. This is used in cases where we want to catch bistability.
        It is an internal method invoked by 'self.updateInitialConditions' along with 'self._generateInitialConditions'

        Input:
        - ICs_dict : Dictionary of ICs as provided by self._generateInitialConditions(curve, par_vals)

        v1.0
        Maurizio De Pitta', Basque Center of Applied Mathematics, Mar 22, 2020
        """
        if self._ngn_setup=='single-astro':
            # In this case the ICs_dict must be in the form of {'v_n': ..., 'v_g': ..., 'gamma_S': ...}
            for k,ics in ICs_dict.items():
                if k in ['v_e','v_n']:
                    if hasattr(ics,'__len__'):
                        ics = np.atleast_1d(ics)
                        assert np.size(ics)==self.pars['N']['e'], "ICs for v of E group must be of same length of N_e"
                    self.network['E'].v = ics
                elif k=='v_g':
                    if hasattr(ics,'__len__'):
                        ics = np.atleast_1d(ics)
                        assert np.size(ics)==self.pars['N']['g'], "ICs for v of G group must be of same length of N_g"
                    self.network['G'].v = ics
                elif k=='gamma_S':
                    if self._glia_to_syn:
                        if hasattr(ics,'__len__'):
                            ics = np.atleast_1d(ics)
                            if np.size(ics)!=self.pars['N']['sx']:
                                # This is the case when ICs are not taken from a previous simulation, but are produced from ex-novo from curves
                                assert np.size(ics)==self.pars['N']['g'], "ICs for gamma_S must be provided of same length of N_g"
                                # Broadcast to size of pars['syn']['nx']['frac']*pars['N']['sx']
                                # Assumes that astrocytes are ordered by unitary increments in the index, that is pars['edges']['sg'][0] = range(N_g)
                                ics = np.repeat(ics,self.pars['N']['sx']//self.pars['N']['g'])
                        # This assignment will copy also scalars and gamma_S passed from a dictionary, with size exactly pars['N']['sx']
                        self.network['Ss'].gamma_S = ics
        elif self._ngn_setup=='model-A':
            # This is similar to the single-astro except that it does not assert same length for ICs
            for k,ics in ICs_dict.items():
                if 'v_' in k:
                    self.network[k.split('_')[-1].upper()].v = ics
                if k=='gamma_S':
                    for sid in self._glia_to_syn.keys():
                        if (not self.__glia_block)and(self._glia_to_syn[sid]):
                            # NOTE: ics in this case must be a scalar in general or exactly an array as long as the number of synapses
                            self.network[sid].gamma_S = ics

    def updateInitialConditions(self,curve, par_vals,par_name='nus',
                                randomize={'e': False, 'i': False, 'g': False, 'gamma': False},
                                ei_only=False,nics_same=1,N_trials=1,return_ICs=False):
        """
        This is a convenient wrapper of _generateInitialConditions, and _updateInitialConditions. For proper debugging
        Invoke the two methods sequentially.
        :param curve:
        :param par_vals:
        :param randomize:

        Return:
        - pvals : array of updated parameter values used in the simulations -- the parameter is dependent on self._ngn_setup

        """

        # This will assure that all relevant keys are passed if any of them was not specified
        rand_default = {'e': False, 'i': False, 'g': False, 'gamma': False}
        randomize = gu.merge_dicts(rand_default,randomize)

        if self._ngn_setup=='single-astrocyte':
            par_name = 'nus' # This is the only parameter allowed in the SAD configuration
        _,ICs = self._generateInitialConditions(curve,par_vals,par_name,randomize=randomize,ei_only=ei_only,nics_same=int(nics_same),N_trials=int(N_trials))
        self._updatedInitialConditions(ICs)
        if return_ICs:
            return ICs

    def _individualICs_from_dict(self,ICs,index):
        if self._ngn_setup=='single-astrocyte':
            ics_keys = ['v_n','v_g','gamma_S']
        else:
            ics_keys = ['v_e', 'v_i','v_g', 'gamma_S']
        assert type(ICs) == dict, "ICs must be a dictionary with model-setup-dependent keys (see self._updateInitialConditions)"
        assert collections.Counter(ICs.keys())==collections.Counter(ics_keys),"ICs dictionary does not have the required keys: " + ics_keys
        individual_ICs = {}
        for k in ics_keys:
            individual_ICs[k] = ICs[k][index]
        return individual_ICs

    def reset(self):
        """
        Reset network parameters and namespaces of network modules to original parameters
        """

        if not self._ngn_setup=='model-A':
            print("'reset' method available only for 'model-A' setup")
            return

        # Replace current parameter with original parameters
        self.pars = cp.deepcopy(self._original_pars)
        # Generate a copy to work on of the original parameters with units
        pars = addParameterUnits(cp.deepcopy(self._original_pars), ngn_setup=self._ngn_setup)

        # Update namespaces
        # network objects are saved in a set NOT a dictionary
        for k,module in [(m.name,m) for m in iter(self.network.objects)]:
            # Cells
            if k in ['E','I','G']:
                if self.pars['N'][k.lower()] > 0:
                    module.namespace = pars['lif'][k.lower()]
                    module.v = pars['lif'][k.lower()]['ICs']

            # Synaptic connections
            if k in ['ee','ie','ei','ii']:
                # Assign synaptic weights
                if k[-1] == 'e':
                    jxx = pars['conn']['j'+k]
                else:
                    jxx = -pars['conn']['j'+k]
                module.namespace = self._synaptic_namespace(pars, k)
                module.jxx = jxx
                # module.delay = pars['syn'][k]['D'] # DISABLED
                # Reinit u0 depending on whether the synapse is under the influence or not of glia
                if (self._glia_to_syn[k])and(not self.__glia_block):
                    module.gamma_S = pars['prer'][k]['ICpre']
                    module.alpha = pars['prer'][k]['xi']
                    module.alpha = pars['syn'][k]['u0']
                else:
                    module.u_0 = pars['syn'][k]['u0']

            # Shared connections from synapses to glia
            if '_g' in k:
                module.wxx = pars['conn']['w'+k[:2]] # The first two characters of the name of the module coincide with the 'xx' synapse identifier
            if 'g_' in k:
                module.namespace = self._gliot_namespace(pars,k)
                module.wsg = pars['conn']['wsg']

            # Gap-junctions
            if k=='gjc':
                module.namespace = {'w': pars['conn']['jgg']}

    @check_units(duration=second, sim_dt=second, rate_dt=second,
                 mon_neu_dt=second,mon_ecs_dt=second,mon_syn_dt=second,
                 transient=second, brp_dt=second)
    def simulate(self,pname=None,pvals=None,
                 stimtype='full',rho_baseline=None,round_decimals=None,
                 duration=0.1*second,sim_dt=0.1*ms,
                 include_ICs=False,ICs=None,
                 mon_neu={'e': [], 'i': [], 'g': []}, mon_neu_dt=0.5*ms,
                 mon_ecs={'e': [], 'i': [], 'g': []}, mon_ecs_dt=0.5*ms,
                 mon_syn={'ee': [], 'ie': [], 'ei': [], 'ii': []}, mon_syn_dt=0.5*ms,
                 mon_rate={'e': True, 'i': False, 'g': False},rate_dt=0.1*ms,
                 mon_spks={'e': False, 'i': False, 'g': False},transient=0.*second,
                 record_spks={'e': True, 'i': True, 'g': True},num_to_record=None,
                 mon_brp={'ee': False, 'ie': False, 'ei': False, 'ii': False},syn_to_record=None,brp_dt=10*ms,save_raw_brp=False,
                 threads=2,
                 safe_cpu=True,
                 build=True,clean=False,
                 enable_debug=False,
                 return_mondict=False):
        """
        Simulate the network. It can be for the given parameter configuration, or for a sequence of parameters values.

        :param pname:
        :param pvals:
        stimtype : 'full' | int number of neurons to stimulate. Could also be an array as long as duration or pvals
        :param duration:
        rho_baseline : {None} | scalar or array For rho values at baseline (only valid if stimtype is rho-timed-array
        round_decimals : {None} | int  Number of decimals to round off to compute dt in the TimeArray (None:
                                       automatically infer from the max of decimals digits in the duration vector
        sim_dt:
        mon_rate:
        mon_spks:
        record_spks : {'e': True,... }  Establish whether or not to record spike timings (True) or only counts (False)
                                        (only relevant if corresponding key in mon_spks is True). Alternatively,
                                        each key may be a 2-element tuple/list of the type: (bool, number_cells_to_record)
        num_to_record : None | {'e': N_e, 'i': N_i, 'g': N_g} Specify the first N_x neurons to record spikes from (only relevant if record_spk[x] is True)
        threads:  int   Provide number of CPUs on which to run the simulation
        safe_cpu: {True} | False  If True always run simulation on CPUs <= max(CPUs) - 1
        build   : {True} | False  Build simulation. If False, allow for multiple calls.
        return_mondict : True | {False}  Return available monitors as dictionaries

        :return:
        """

        # Brian 2 settings
        prefs.devices.cpp_standalone.openmp_threads = safe_threads(threads,safe_cpu)
        if enable_debug:
            prefs.logging.file_log = True
            prefs.logging.file_log_level = 'DEBUG'
            prefs.logging.save_script = False
        else:
            prefs.logging.file_log = False
            prefs.logging.delete_log_on_exit = True
        defaultclock.dt = sim_dt

        # Delete

        # Modules names (does NOT include monitors if called on a single run)
        modules = [m.name for m in iter(self.network.objects)]

        device.delete(force=True)      # Default settings for monitors: No monitors
        if self._nea_setup=='l5e+ecs':
            # Monitor settings (only neuronal, ecs and synaptic monitors allowed)
            mon_neu_default = {'e': ['v','N_i','K_i','C_i']}
            mon_ecs_default = {'e': ['N_e','K_e','C_e']}
            mon_syn_default = {'ee': ['Nt_s']}
            mon_rate_default = {'e': False, 'i': False, 'g': False}
            mon_spks_default = {'e': False, 'i': False, 'g': False}
            num_to_record_default = {'e': self.pars['N']['e']}
            syn_to_record_default = {}


        # elif self._nea_setup=='single-astro':
        #     # Monitors' settings
        #     mon_rate_default = {'e': False, 'g': False}
        #     mon_spks_default = {'e': False, 'g': False}
        #     mon_brp_default = {}
        #     num_to_record_default = {'e': self.pars['N']['e'], 'g': self.pars['N']['g']}
        #     syn_to_record_default = {}
        #     mon_keys = ['e','g']
        #     # DEBUG
        #     # gamma_mon = StateMonitor(self.network['Ss'],variables=['gamma_S'],record=np.arange(10))
        # elif self._ngn_setup=='model-A':
        #     # TODO: Stimulus settings
        #     # Monitor setting
        #     mon_rate_default = {'e': False, 'i': False, 'g': False}
        #     mon_spks_default = {'e': False, 'i': False, 'g': False}
        #     mon_brp_default = {'ee': False, 'ie': False, 'ei': False, 'ii': False}
        #     dt_rate_mon = {'e': 0.1*ms, 'i': 0.1*ms, 'g': 0.1*ms}
        #     num_to_record_default = {'e': self.pars['N']['e'], 'i': self.pars['N']['i'], 'g': self.pars['N']['g']}
        #     syn_to_record_default = dict(zip(['ee','ie','ei','ii'],[0,0,0,0]))
        #     mon_keys = ['e', 'i', 'g']

        # Update monitors with custom-given options
        mon_neu = gu.merge_dicts(mon_neu_default,mon_neu) if mon_neu != None else mon_neu_default
        mon_ecs = gu.merge_dicts(mon_ecs_default,mon_ecs) if mon_ecs != None else mon_ecs_default
        mon_syn = gu.merge_dicts(mon_syn_default,mon_syn) if mon_syn != None else mon_syn_default
        mon_rate = gu.merge_dicts(mon_rate_default,mon_rate) if mon_rate != None else mon_rate_default
        mon_spks = gu.merge_dicts(mon_spks_default,mon_spks) if mon_spks != None else mon_spks_default
        # mon_brp = gu.merge_dicts(mon_brp_default, mon_brp) if mon_brp!=None else mon_brp_default
        # # Adjust mon_brp: if 'e' or 'i' keys are specified, automatically set to that value the 'xe' ('xi') synapses
        # for k in ['e','i']:
        #     if k in mon_brp:
        #         for x in ['e','i']:
        #             mon_brp[x+k] = mon_brp[k]
        #         # Delete 'e' or 'i' key and deals with individual synapse types
        #         mon_brp.pop(k)
        # Adjust mon_syn: if 'e' or 'i' keys are specified, automatically set to that value the 'xe' ('xi') synapses
        for k in ['e', 'i']:
            if k in mon_syn:
                for x in ['e', 'i']:
                    mon_syn[x+k] = mon_syn[k]
                # Delete 'e' or 'i' key and deals with individual synapse types
                mon_syn.pop(k)

        # # TODO: Number of cells and synapses to record: Only relevant once the Parameters are finalized
        # # Number of neurons to record
        # if (num_to_record!=None):
        #     assert (type(num_to_record)==type({}))and(all(item in num_to_record_default.keys() for item in num_to_record.keys())),"num_to_record must be a dictionary with keys in "+num_to_record_default.keys()
        # num_to_record = gu.merge_dicts(num_to_record_default,num_to_record) if num_to_record != None else num_to_record_default
        #
        # # Set the max number of synapses of each type to record per recorded neuron
        # if (syn_to_record!=None):
        #     assert (type(syn_to_record)==type({}))and(all(item in syn_to_record_default.keys() for item in syn_to_record.keys())),"syn_to_record must be a dictionary with keys in "+syn_to_record_default.keys()
        # syn_to_record = gu.merge_dicts(syn_to_record_default,syn_to_record) if syn_to_record != None else syn_to_record_default
        # # Adjust syn_to_record if 'e' or 'i' keys are specified, identifying the number of 'xe' ('xi') synapses based on the ratio in the network
        # for k in ['e','i']:
        #     # The following is only for the case where syn_to_record are specified by 'e' and 'i'
        #     if k in syn_to_record:
        #         nksyn_total = self.pars['N']['e'+k]+self.pars['N']['i'+k]
        #         # Compute fraction of 'ek' synapses
        #         fraction = self.pars['N']['e'+k]/nksyn_total
        #         syn_to_record['e'+k] = int(np.round(fraction*syn_to_record[k]))
        #         # The complementary synapses 'ik' are the specified 'k' synapses - the number of 'ek' synapses
        #         syn_to_record['i'+k] = syn_to_record[k] - syn_to_record['e'+k]
        #         # Delete 'e' or 'i' key and deals with individual synapse types
        #         syn_to_record.pop(k)

        # TODO: Add rates and spike monitors for
        for k, vars in mon_neu.items():
            if (len(vars)>0) and (k.upper() in modules) and (self.pars['N'][k]>0):
                self.network.add(StateMonitor(self.network[k.upper()],variables=vars,record=True,name=k.upper()+'_neumon'),dt=mon_neu_dt)
        for k, vars in mon_ecs.items():
            if (len(vars)>0) and (k.upper() in modules) and (self.pars['N'][k]>0):
                self.network.add(StateMonitor(self.network[k.upper()],variables=vars,record=True,name=k.upper()+'_ecsmon'),dt=mon_neu_dt)
        for k, vars in mon_syn.items():
            if (len(vars)>0) and (k.upper() in modules) and (self.pars['N'][k]>0):
                self.network.add(StateMonitor(self.network[k.upper()],variables=vars,record=True,name=k.upper()+'_synmon'),dt=mon_neu_dt)
        # TODO: Rate and Spike Monitors
        # # Add monitors (only for cells)
        # for k, flag_mon in mon_rate.items():
        #     # Only add if the number N_k>0
        #     if flag_mon and (k.upper() in modules) and (self.pars['N'][k] > 0):
        #         if self.pars['geom']['N_clusters']<2:
        #             self.network.add(PopulationRateMonitor(self.network[k.upper()], name=k.upper()+'_poprate'))
        #         else:
        #             for cn,ci in enumerate(self.pars['geom'][k+'_cluster']):
        #                 # Because subgroups require at the moment only contiguous indexes, you must use ci[0]:ci[-1]
        #                 # TODO: does not check whether ci is of size 1: in this case it might not work
        #                 self.network.add(PopulationRateMonitor(self.network[k.upper()][ci[0]:ci[-1]], name=k.upper() + '_poprate_'+str(cn)))
        # for k, flag_mon in mon_spks.items():
        #     # Only add if the number N_k>0
        #     if flag_mon and (k.upper() in modules) and (self.pars['N'][k] > 0):
        #         if transient>0:
        #             # Simple error checking: you cannot estimate count correctly if you don't record all spikes with transient>0
        #             assert record_spks[k],"Spike count cannot be estimated correctly when transient>0 and mon_spks['"+k+"'] is False: set it to True instead"
        #         # Specify the first num_to_record[k] cells to record spikes from (default: all cells will be recorded)
        #         if self.pars['geom']['N_clusters']<2:
        #             # 'geom' is only specified in the ngn setup, but this assignment is required also for single-astro configurations
        #             self.network.add(SpikeMonitor(self.network[k.upper()][:int(num_to_record[k])], name=k.upper()+'_spikes', record=record_spks[k]))
        #         else:
        #             for cn,ci in enumerate(self.pars['geom'][k+'_cluster']):
        #                 # TODO: does not check whether ci is of size 1: in this case it might not work
        #                 # Indexes of selected cells to record spikes from
        #                 # Because subgroups require at the moment only contiguous indexes, you must use nsel[0]:nsel[-1]
        #                 nsel = ci[:int(num_to_record[k])]
        #                 self.network.add(SpikeMonitor(self.network[k.upper()][nsel[0]:nsel[-1]], name=k.upper() + '_spikes_'+str(cn), record=record_spks[k]))

        # Effective simulation
        if pname==None:
            # Run the network as it is
            self.network.run(duration,report='text')
        else:
            # TODO: This handles the case of multiple simulations, when the parameter is given as an array of values
            # TODO: Refer to network_classes_brian same field
            assert hasattr(pvals,'__len__'),"pvals must be an array-like type if pname!=None"

        # Code builder
        if build:
            # Build at the end
            device.build(directory=self._code_dir,clean=clean)
            self.__build = True

            if return_mondict:
                # Retrieve update list of modules names (including monitors)
                modules = [m.name for m in iter(self.network.objects)]
                monitors = [c.upper() + '_' + mt + 'mon' for c in mon_keys for mt in ['neu', 'ecs' , 'syn']]
                mons = {'neu': {}, 'ecs': {}, 'syn': {}}

                # # Build monitor names (rate and spks type -- brp is handled separately below)
                # if self.pars['geom']['N_clusters']<2:
                #     monitors = [c.upper()+'_'+mt for c in mon_keys for mt in ['spikes','poprate']]
                # else:
                #     monitors = [c.upper() + '_' + mt + '_' + str(cn) for c in mon_keys for mt in ['spikes', 'poprate'] for cn in range(self.pars['geom']['N_clusters'])]
                # mons = {'rate': {}, 'spks': {}, 'brp': {}}
                # if self.pars['geom']['N_clusters']>=2:
                #     for k in mons.keys():
                #         mons[k] = {name[0].lower(): {} for name in monitors}

                for name in monitors:
                    if name in modules:
                        if ('mon' in name):
                            if 'neu' in name:
                                mons['neu'][name[0].lower()] = bu.monitor_to_dict(self.network[name],monitor_type='state',transient=transient/second)
                            elif 'ecs' in name:
                                mons['neu'][name[0].lower()] = bu.monitor_to_dict(self.network[name],monitor_type='state',transient=transient/second)
                            elif 'syn' in name:
                                mons['neu'][name[0].lower()] = bu.monitor_to_dict(self.network[name],monitor_type='state',transient=transient/second)
                            else:
                                print('WARNING: No StateMonitor specified for this configuration')
                                pass
                        # # TODO: Update the following code for other types of monitors, when introduced
                        # elif ('poprate' in name):
                        #     if self.pars['geom']['N_clusters']<2:
                        #         mons['rate'][name[0].lower()] = bu.monitor_to_dict(self.network[name], monitor_type='poprate', window='flat',width=rate_dt,transient=transient/second)
                        #     else:
                        #         # Retrieve cluster index
                        #         ci = index_from_string(name)
                        #         mons['rate'][name[0].lower()][ci] = bu.monitor_to_dict(self.network[name], monitor_type='poprate', window='flat',width=rate_dt,transient=transient/second)
                        # elif 'spikes' in name:
                        #     if not record_spks[name[0].lower()]:
                        #         if self.pars['geom']['N_clusters']<2:
                        #             # Only counts
                        #             mons['spks'][name[0].lower()] = bu.monitor_to_dict(self.network[name], monitor_type='spike', fields=['count'], transient=transient/second)
                        #         else:
                        #             # Retrieve cluster index
                        #             ci = index_from_string(name)
                        #             mons['spks'][name[0].lower()][ci] = bu.monitor_to_dict(self.network[name], monitor_type='spike', fields=['count'], transient=transient/second)
                        #     else:
                        #         if self.pars['geom']['N_clusters']<2:
                        #             # Counts and spikes
                        #             mons['spks'][name[0].lower()] = bu.monitor_to_dict(self.network[name], monitor_type='spike', fields=['spk', 'count'],transient=transient/second)
                        #         else:
                        #             # Retrieve cluster index
                        #             ci = index_from_string(name)
                        #             mons['spks'][name[0].lower()][ci] = bu.monitor_to_dict(self.network[name], monitor_type='spike', fields=['spk', 'count'], transient=transient/second)

                # TODO: Currently Removed BRP (synaptic?) monitor handling

                # Clean of monitors
                mons = gu.dict_stripper(gu.dict_stripper(mons))

                return mons
            else:
                device.delete(force=True)

#-----------------------------------------------------------------------------------------------------------------------
# Testing classes
#-----------------------------------------------------------------------------------------------------------------------
if __name__=="__main__":
    # Single Neuron configuration
    nea = NEAGroup(N_e=1, N_i=0, N_g=0,
                   nea_setup='l5e+ecs',
                   ne_pars = {}, ni_pars = {}, ng_pars = {}, ecs_pars = {},
                   ee_pars = 'default', ie_pars = 'same', ii_pars = 'default', ei_pars = 'same',
                   ex_pars = {}, ix_pars = 'same', gx_pars = {},
                   network_connectivity=False,
                   network_geometry = None, network_topology = {}, N_clusters = 0,
                   code_dir = './codegen')
    # Simulation
    mons = nea.simulate(duration=0.1*second,sim_dt=0.1*us,
                 mon_neu={'e': [], 'i': [], 'g': []}, mon_neu_dt=0.5*ms,
                 mon_ecs={'e': [], 'i': [], 'g': []}, mon_ecs_dt=0.5*ms,
                 mon_syn={'ee': [], 'ie': [], 'ei': [], 'ii': []}, mon_syn_dt=0.5*ms,
                 threads=2,
                 safe_cpu=True,
                 build=True,clean=True,
                 enable_debug=False,
                 return_mondict=True)

    # Visualization
    fig,axs = plt.subplots(4,1)
    # axs[0].plot(mons['neu']['t'],mons['neu']['v'],'k-')
    axs[1].plot(mons['neu']['t'],mons['neu']['v'],'k-')
    axs[2].plot(mons['ecs']['t'],mons['ecs']['N_e'],'g-')
    axs[3].plot(mons['ecs']['t'],mons['ecs']['Nt_s'],'y-')

    #-----------------------------------------------------------------------------------------------------------------------
    # Visualization
    #-----------------------------------------------------------------------------------------------------------------------
    plt.show()

