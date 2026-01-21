"""
ngn_topology.py

Contains several routines strictly related to generate NG networks of specified topology.
This module lumps together all topology-related routines originally present in ngn_model.py

v1.1 -- Extended for NEA setups (besides NGN)
Maurizio De Pitta', Krembil Brain Institute, January 2026

v1.0
Maurizio De Pitta', Basque Center for Applied Mathematics, October 24, 2019
"""

import cython
import numpy as np
import numpy.ma as ma # Masked array
import scipy.sparse as sparse
import freud # Provides Voronoi 2D methods in closed box
import scipy.spatial as space
import matplotlib.pylab as plt
import matplotlib.colors as mplc
import matplotlib.patches as mpatches

# NetworkX necessary to build the actual network
import networkx as nx
# Module to correctly build bipartite graphs
from networkx.algorithms import bipartite as bip
import sklearn.neighbors as nbors
import functools as funct
import mpl_toolkits.mplot3d as plt3d
import mpl_toolkits.mplot3d.art3d as art3d
# Maybe to visualize it use graph-tools
# see it here: https://graph-tool.skewed.de/

# Import custom modules
import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'),'Dropbox/Ongoing.Projects/pycustommodules'))
import general_utils as gu
import graphics_utils.plot_utils as pu

#-----------------------------------------------------------------------------------------------------------------------
# ROUTINES TO CHECK CONNECTIONS
#-----------------------------------------------------------------------------------------------------------------------
def calculate_degree(edges,direction='out'):
    if direction=='in':
        nodes,counts = np.unique(edges[-1],return_counts=True)
        print(direction + '\t',np.size(nodes),'\t',np.mean(counts),'\t',np.std(counts))
    elif direction=='out':
        nodes,counts = np.unique(edges[0],return_counts=True)
        print(direction + '\t',np.size(nodes),'\t',np.mean(counts),'\t',np.std(counts))
    else:
        calculate_degree(edges,direction='in')
        calculate_degree(edges,direction='out')

def total_synapse_number(N_e, N_i, prob_conn):
    """
    Provide a basic estimation of the total number of synapses in the network

    Input arguments:
    - N_e : Number of E neurons
    - N_i : Number of I neurons
    - prob_conn: dictionary of probabilities for connection type

    Return:
    - total number of synapses (int)
    """
    C_ee = prob_conn['ee']*N_e**2
    C_ie = prob_conn['ie']*N_e*N_i
    C_ei = prob_conn['ei']*N_i*N_e
    C_ii = prob_conn['ii']*N_i**2
    return int(C_ee + C_ei + C_ie + C_ii)

def synapse_coordinates(cds_pre,cds_post,edges,location='post'):
    """
    Provide coordinates for location of synapses.

    Input parameters:
    - cds_pre  : coordinates of presynaptic cells [ncells x nfeatures]
    - cds_post : coordinates of postsynaptic cells [ncells x nfeatures]
    - edges    : connections between pre and post as any of edges[k] dictionary entry from connect
    - location : string   Whereto locate synapses
      *pre  : synapses concides with presynaptic cell location
      *post : synapses concides with postsynaptic cell location

    Return:
    - coordinates : array (nsyn x nfeatures)
    """

    if (np.size(cds_pre)>0)and(np.size(cds_post)>0):
        if location=='post':
            cds_syn = cds_post[edges[-1]]
        elif location=='pre':
            cds_syn = cds_pre[edges[0]]
    else:
        cds_syn = np.zeros(np.shape(cds_pre))

    # The atleast_2d is to assures that even if coordinates are empty, they can be referenced as coordinates
    return np.atleast_2d(cds_syn)

def retrieve_true_edges(sym_edges,ni_pre,ni_pst):
    """
    Convert symbolic edges to true edges (used in the random-clustered network)

    Input:
    - sym_edges : symbolic edges
    - ni_pre    : indexes of pre-synaptic neurons
    - ni_pst    : indexes of post-synaptic neurons

    Return:
    - edges : 2xN array of true edges
    """

    N_edges = np.shape(sym_edges)[1]
    pre,pst = np.zeros(N_edges),np.zeros(N_edges)
    for i,i_pre in enumerate(ni_pre):
        pre[sym_edges[0]==i] = i_pre
    for j,i_pst in enumerate(ni_pst):
        pst[sym_edges[1]==j] = i_pst

    return np.vstack((pre,pst)).astype(np.int32)

def retrieve_cluster_indexes(id_key,edges,geom):
    """
    Retrieve indexes of edges by cluster. This is needed for proper plotting of different clusters.

    Input
    :param id_key:
    :param edges:
    :param geom:
    :return:
    """
    Nc_pre = len(geom[id_key[0]+'_cluster'])
    Nc_pst = len(geom[id_key[-1]+'_cluster'])
    # List of indexes of individual clusters in the edge
    indexes = []
    for i in range(Nc_pre):
        # Identify all neurons in the cluster pre-synaptically
        ni_pre = geom[id_key[0]+'_cluster'][i]
        i_pre = np.where(np.isin(edges[id_key][0],ni_pre))[0]
        for j in range(Nc_pst):
            # Identify all neurons in the cluster post-synaptically
            ni_pst = geom[id_key[-1]+'_cluster'][j]
            i_pst = np.where(np.isin(edges[id_key][-1],ni_pst))[0]
            # Find overlap (this is possible because each neuron pair is unique)
            i_clu = np.isin(i_pre,i_pst)
            indexes.append(i_pre[i_clu])
            # The real neurons are (pre) edges[k][0][index[i]], (post) edges[k][-1][index[i]]

    return indexes

#-----------------------------------------------------------------------------------------------------------------------
# ROUTINES TO BUILD CONNECTIONS
#-----------------------------------------------------------------------------------------------------------------------
@cython.compile
def conn_matrix_synastro(N1,N2,w):
    """
    Generate a matrix of connections between synapses and astrocytes.
    This matrix is generally huge (as the number of synapses is large) and it is sparse.

    Input arguments:
    - N1 : int    Number of synapses
    - N2 : int    Number of glia
    - w  : float  weight of synaptic connections (currently just taken identical for all connections)

    Return:
    - connectivity_matrix : a csr sparse matrix with non-zero connections for syn-glia pairs
    """
    rows = np.arange(N1)
    cols = (np.floor(np.random.permutation(N1)/float(N1)*N2)).astype(int)
    W = w*np.ones(N1)
    conn_matrix = sparse.csr_matrix((W, (rows, cols)), shape=(N1, N2), dtype=int)
    return conn_matrix

def connect(N1,N2=None,topology='random',
            prob=None,seed=None,
            spatial_geometry={'id': None},
            clusters=None,return_cluster_index=False,
            cds_neurons=None,syn_neu_indexes=None,
            cds_glia=None):
    """
    Define connectivity matrix of N1 x N2 nodes according to topology.

    Input arguments:
    - N1       :  int    Size of source nodes
    - N2       :  int    Size of target nodes. If None automatically N2==N1
    topology:
    - random : E-R graph with probability "prob" (avilable also in the *-clustered version)
    - random-const-outdeg : random network with fixed out degree (computed by prob*N1 or N2) (avilable also in the *-clustered version)
    - all-to-one : used for single astrocyte configuration
    - random-clustered : randomly-connected clustered (requires specification of 'clusters')
    * s2g-random : synapse-to-glia random connectivity with probability to pick synapses prob (1 if prob==None)
    * s2g-random-const-outdeg : synapse-to-glia random connectivity but with fixed number of synapses per astrocyte. Synapses are picked with probability prob (1 if pob==None)
    - prob        : float in [0,1]   Probability of connection in the case of random networks
    - spatial_geometry : {'id': *, 'dmax': *, radius': *} (This dictionary is usually provided inside ngn_parameters)
      -- 'id'     : {None} | 'planar' | 'spherical'
      -- 'glia_dist' : max distance between glial cells for connections in [um]
      -- 'dsize'     : float  Characteristic size of the geometry (used only if 'id': 'spherical' :: this carries info on the sphere radius in [um]
    - clusters : list of [cluster-id,pre-neurons-array,post-neurons-array]  Must be specified only if topology='random-clustered'
    - cds_neurons : neuronal cells coordinates (provided as arrays of [N_cells x ndims]
    - cds_glia    : glial cells coordinates (provided as arrays of [N_cells x ndims]

    Return:
    - edges : return edges list (only if return_edges=True)

    """

    # Allocation of empty outputs
    # Edges are saved in np.uint32 format
    edges = np.zeros((2,0))
    cluster_indexes = np.zeros(0,)  # atleast_1-d empty array --> it will be filled with clustering indexes if 'clustering' is detected

    # Case of no topology (no connections)
    if topology==None:
        return edges if not return_cluster_index else edges,cluster_indexes

    # Initialize random generator
    np.random.seed(seed)

    # Pre-process input data
    N1 = int(N1)
    if N2!=None :
        N2 = int(N2)
        # Either N1 or N2 are <=0 then no connections
        # Assert type and handle N2
        if N1*N2 <= 0:
            return edges if not return_cluster_index else edges,cluster_indexes

    # Unfold spatial geometry:
    geometry = spatial_geometry['id']
    if geometry!=None:
        # The following variables are used only when geometry is specified
        gdist = spatial_geometry['glia_distance']
        radius = spatial_geometry['dsize']

    if topology=='random':
        assert (prob!=None)and(prob>=0.)&(prob<=1.0), "probability of connections (arg) must be in [0,1]"
        if prob>0.:
            # It is easier to think of a bipartite graph starting from target and going "backward" to source cells, and then flip the edges
            if N2 is None:
                # This treats the case of recurrent connections
                edges = np.atleast_2d(nx.edges(nx.fast_gnp_random_graph(N1,p=prob,seed=seed, directed=True))).T.astype(dtype=np.uint32)[::-1]
            else:
                # This is the classic erdos-renyi network based on a bipartite graph algorithm
                # edges = np.atleast_2d(nx.edges(bip.random_graph(N2,N1,p=prob,seed=seed,directed=True))).T.astype(dtype=np.uint32)[::-1]
                edges = np.atleast_2d(nx.edges(bip.random_graph(N2,N1,p=prob,seed=seed,directed=False))).T.astype(dtype=np.uint32)[::-1]
                # Adjust indexes (because bip.random_graph produces post/target indexes from N_pre:(N_pre+N_post-1)
                edges[0] -= N2
    elif topology=='random-const-outdeg':
        # This case corresponds to fixed number of connections
        # arg in this case reflects the exact number of connections
        assert (prob!=None)and((prob>=0.)and(prob<=1.0)), "probability of connections (arg) must be in [0,1]"
        if prob>0.:
            if N2 is None:
                degree = np.rint(prob*N1).astype(np.uint32)
                pre_cells = np.arange(N1)
                edges = np.vstack((np.concatenate([np.random.choice(pre_cells[pre_cells!=n],size=np.atleast_1d(degree),replace=False) for n in range(N1)]),
                                   np.concatenate(np.tile(np.arange(N1), (degree, 1)).T)))
            else:
                degree = np.rint(prob*N1).astype(np.uint32)
                pre_cells = np.arange(N1)
                edges = np.vstack((np.concatenate([np.random.choice(pre_cells,size=np.atleast_1d(degree).astype(np.uint32),replace=False) for n in range(N2)]),
                                   np.concatenate(np.tile(np.arange(N2), (degree, 1)).T)))
    elif topology=='all-to-one':
        # All-to-one connections: This assumes that N2 is 1
        edges = np.vstack((np.arange(N1),np.ones(N1)))
    # Handling of clustered topology for E-I neurons only
    elif ('clustered' in topology)and(all(x not in topology for x in ['s2g','g2s'])):
        # TODO: Warning this section has been tested for g2g connections only when g2g-random-clustered is specified (hard-wired missing spatial_geometry)
        assert len(clusters)==3,"clusters must be specified as a list or tuple of 3 elements: cluster-id string + 2 array-like of indexes for pre and post neurons respectively"
        cluster_topology = topology.split('-clustered')[0]
        # This is the instance that uses cluster_indexes
        for i,ni_pre in enumerate(clusters[1]):
            # ci: cluster index; ni: neuron index pre-synaptic neurons
            if np.size(ni_pre)>0:
                for j,ni_pst in enumerate(clusters[-1]):
                    if np.size(ni_pst)>0:
                        # Consider "symbolic edges" build by recursive call to the function in the "random" configuration
                        N1_ = np.size(ni_pre)
                        N2_ = np.size(ni_pst)
                        # The following check avoid autapses
                        if (i==j)and(clusters[0] in ['ee','ii','gg']): N2_ = None
                        # G-G connections are bidirectional therefore we only need the diagonal and the upper/lower elements (here we pick up the upper ones also satisfying N1 as source nodes and N2 as target nodeu)
                        # ONGOING PROGRESS
                        print('Generating connectivity for '+clusters[0].upper()+' : cluster '+str(i)+','+str(j))
                        if clusters[0]=='gg':
                            if j>=i:
                                edges_ = connect(N1_,N2=N2_,topology=cluster_topology,prob=prob[i,j],seed=seed,spatial_geometry={'id':None},clusters=None)
                            else:
                                edges_ = np.zeros((2,0))
                        else:
                            # In the case of all other connections all connections between the different clusters are legitimate
                            edges_ = connect(N1_,N2=N2_,topology=cluster_topology,prob=prob[i,j],seed=seed,spatial_geometry={'id': None},clusters=None)
                        # Replace symbolic edges by true cell indexes
                        true_edges = retrieve_true_edges(edges_,ni_pre,ni_pst)
                        edges = np.concatenate((edges,true_edges),axis=1)
                        # Updated cluster indexes: clusters are numbered in row-major order as (i*nrows)+j. You must think of synapses type as: [[A0-->B0,A0-->B1,...,A0-->BN],[A1-->B0 A1-->B1,...,A1-->BN],...]
                        # This information must then be passed the proper astrocyte population (in ngn_connections)
                        cluster_indexes = np.concatenate((cluster_indexes,(i*len(clusters[1])+j)*np.ones(np.shape(true_edges)[1])))
    elif topology=='s2g-random':
        assert (N2!=None)and(N2>0),"Number of glial cells must be >0"
        if prob == None: prob = 1.
        N1_ = int(np.rint(prob*N1))  # effective number of synapses to be connected
        if N1_<N2 : print("WARNING: Number of glial cells should be >= number of synapses to connect (p*N1)")
        edges = np.vstack((np.arange(N1_),np.random.randint(0,N2,size=N1_)))
    elif topology=='s2g-random-const-outdeg':
        # NOTE: if N1<N2 (by the choice of prob of connection) the actual random assignment is partial and the situation should not be considered.
        # Generally you should consider N1>=N2
        assert (N2!=None)and(N2>0),"Number of glial cells must be >0"
        if prob==None: prob=1.
        N1_ = np.rint(prob*N1).astype(np.uint32) # Effective number of connected synapses
        if N1_<N2: print("WARNING: Number of glial cells should be <= number of synapses to connect (p*N1)")
        degree = N1_//N2
        # We need to make sure that we have as many glia indexes as N1 (the number of synapses)
        if np.mod(N1_,N2)==0:
            # This will assure that connections are uniform for each glial cell
            glia_indexes = (np.tile(np.arange(N2), (1,degree)))[0]
        else:
            glia_indexes = (np.tile(np.arange(N2), (1,degree+1)))[0,:N1_]
        # Shuffle glia indexes
        np.random.shuffle(glia_indexes)
        edges = np.vstack((np.random.choice(np.arange(N1), size=N1_, replace=False), glia_indexes))
    elif topology=='g2g-random':
        # This is for glia-to-glia connections. Is a dual random network: in the sense that connection between i--j is duplicated:
        # i-->j and j--i: for this reason we use a simple undirected graph
        assert (prob!=None)and(prob>=0.)&(prob<=1.0), "probability of connections (arg) must be in [0,1]"
        if prob>0.:
            if N2 is None:
                # This treats the case of recurrent connections
                edges = np.atleast_2d(nx.edges(nx.fast_gnp_random_graph(N1,p=prob,seed=seed,directed=False))).T.astype(dtype=np.uint32)[::-1]
            else:
                # This is the classic erdos-renyi network based on a bipartite graph algorithm
                # edges = np.atleast_2d(nx.edges(bip.random_graph(N2,N1,p=prob,seed=seed,directed=True))).T.astype(dtype=np.uint32)[::-1]
                edges = np.atleast_2d(nx.edges(bip.random_graph(N2,N1,p=prob,seed=seed,directed=False))).T.astype(dtype=np.uint32)[::-1]
                # Adjust indexes (because bip.random_graph produces post/target indexes from N_pre:(N_pre+N_post-1)
                edges[0] -= N2
    # The following are options only available in the spatial network case
    elif (topology=='s2g-by-domain')and(geometry!=None)and(prob>0.):
        # This connect all synapses of a neuron to the associated glial domain
        N1_ = np.rint(prob*N1).astype(np.uint32)  # Effective number of synapses to be connected with glia
        glial_indexes = find_cellular_domains(cds_glia.T, cds_neurons.T,geometry=geometry,radius=radius)
        glial_indexes = glial_indexes[syn_neu_indexes]
        syn_indexes = np.random.choice(np.arange(N1), size=N1_, replace=False)
        edges = np.vstack((syn_indexes,glial_indexes[syn_indexes]))
    # Handling of synapse-to-glia clustered connections
    elif all(x in topology for x in ['clustered','s2g'])and(np.any(prob>0.)):
        assert len(clusters)==3,"clusters must be specified as a list or tuple of 3 elements: cluster-id string + 2 array-like of indexes for input synapses and glia respectively"
        cluster_topology = topology.replace('-clustered','')
        # Extract synaptic clusters information
        id_cluster,syn_count = np.unique(clusters[1],return_counts=True)
        N_clusters = len(clusters[-1])
        if np.size(prob)>1:
            # Flatten probability if needed when probability is an array of size N_clusters)
            prob_ = np.ravel(prob,order='C')    # Synaptic/glia clusters are counted in row-major fashion (see 'clustered' for E/I neurons above)
        else:
            # If probability is a scalar then generate a probability of size id_cluster
            prob_ = prob*np.ones(np.size(id_cluster))
        # syn_clusters = [[i]*c for i,c in zip(id_cluster,syn_count)]
        n0 = 0
        # Extract diagonal information (needed for 'within' topology
        if any([ks in topology for ks in ['within','chained']]):
            diag_idx = np.diag(np.reshape(id_cluster,(N_clusters,N_clusters),order='C'))
            if 'chained' in topology:
                neighbors = 2 # Fixed number of chained consecutive clusters
                true_syn_indexes_all = [np.arange(cnt) for cnt in syn_count]
        for i in id_cluster:
            # Check glia cluster size (and assign synapses only if there are glial cells in the designated cluster)
            # In general you have N_cluster**2 of synapses and N_clusters of glia. Each column in the block matrix of the synapses tells what synapses should be associated with the respective glia cluster
            if 'forward' in topology:
                # Glia attached to in-cluster and outgoing synapses
                cidx = i//N_clusters
            elif 'backward' in topology:
                # Glia attached to in-cluster and ingoing synapses
                cidx = i%N_clusters
            else:
                # Default is 'forward'
                cidx = i//N_clusters
            if np.size(clusters[-1][cidx])>0:
                # clustered-within topology implies that synaptic connections to glia are only for synapses within the cluster and not for those between different clusters
                if ('within' in topology) and (i not in diag_idx):
                    # It will skip the current configuration, but increment the positional index for the synapses
                    n0 += syn_count[i]
                    continue
                if ('chained' in topology) and (i not in np.union1d(diag_idx,np.r_[diag_idx[:-1]+1,N_clusters*(N_clusters-1)])):
                    # It will skip the current configuration, but increment the positional index for the synapses
                    n0 += syn_count[i]
                    continue
                # cluster_topology = (topology.split('-clustered', ''))[0]
                cluster_topology = topology.split('-clustered')[0]
                N1_ = syn_count[i]             # Number of synapses in the cluster
                N2_ = np.size(clusters[-1][cidx]) # Number of glia in the associated cluster -- recall that synaptic clusters are in row-major order
                edges_ = connect(N1_,N2=N2_,topology=cluster_topology,prob=prob_[i],seed=seed,spatial_geometry={'id': None},clusters=None)
                # Replace symbolic edges by true synaptic indexes
                true_syn_indexes = n0 + edges_[0]
                # Provide true glial indexes
                true_glia_indexes = clusters[-1][cidx]
                glia_indexes = np.zeros(np.shape(edges_)[1])
                for j, i_glia in enumerate(true_glia_indexes):
                    glia_indexes[edges_[1]==j] = i_glia
                # Provide effective edges: you only need a for cycle in this case on glia, as synaptic contacts are numbered, and unique.
                # In this fashion you avoid a lengthy for cycle on synaptic indexes and you do not need to call 'retrieve_true_edges'
                true_edges = np.vstack((true_syn_indexes, glia_indexes)).astype(np.int32)
                edges = np.concatenate((edges,true_edges),axis=1)
                # Updated offset for synaptic indexes
                n0 += N1_
    elif (topology=='g2g-nearest')and(geometry!=None)and(prob>0.):
        # This is only for glia-to-glia recurrent connections
        if geometry=='planar':
            # The graph can be taken undirected in this case, since a connection i-j is also j-i for gap junction nature
            edges = np.atleast_2d(nx.edges(nx.random_geometric_graph(N1, radius=gdist, dim=2, pos=cds_glia, p=2, seed=seed))).T.astype(dtype=np.uint32)
        elif geometry=='spherical':
            # In this case we use a KDTree finding neighbors within a radius. Because we work with haverside metrics however,
            # we first need lat and long of the glial cells locations
            assert radius > 0., "Sphere radius must be specified and >0"
            metric = nbors.DistanceMetric.get_metric('haversine')
            # This metric also requires the points to be given in lat,long
            gp_ = provide_lat_long(cds_glia[:, 0],cds_glia[:, 1],cds_glia[:, 2], radius=radius)
            celldom = nbors.KDTree(gp_,metric=metric)
            glia_indexes = celldom.query_radius(gp_,r=gdist,count_only=False,return_distance=False)
            # Generate the (redundant) edges (each cell is associated with its neighbors and so we count connections at least twice)
            edges_pre = np.concatenate([i*np.ones(np.size(gi)) for i,gi in enumerate(glia_indexes)])
            edges_pst = np.concatenate(glia_indexes)
            # Get only unique pairs
            edges = np.unique(np.c_[edges_pre,edges_pst],axis=1)
        # Sort out only N1_ connections based on probability of connections (relevant only if p<1)
        if prob<1.0:
            N_gjs = int(np.shape(edges)[1])
            N1_ = np.rint(prob*N_gjs).astype(np.uint32)  # Effective number of gap junctions
            edges = edges[:,np.random.choice(np.arange(N_gjs), size=N1_, replace=False)]
    if return_cluster_index:
        return edges.astype(np.int32),cluster_indexes.astype(np.int32)
    else:
        return edges.astype(np.int32)

def edges_list(conn_matrix,x_shift=0,y_shift=0):
    """
    Provide a description of connectivity in terms of (x,y) pairs. Optionally shift x,y coordinates according to block
    position of conn_matrix within the whole network connectivity matrix

    Input arguments:
    - conn_matrix  : Connectivity matrix (binary)
    - x_shift      : Integer shift on x corrdinate
    - y_shift      : Integer shift on y corrdinate

    Return :
    - coords       : Array of integers of n x 2 size with all connection pairs
    """
    coords = np.transpose(conn_matrix.nonzero())
    coords[:,0] += x_shift
    coords[:,1] += y_shift
    # intc type is correctly read by C routines
    return coords.astype(np.intc)

def ngn_connections(N_e,N_i,N_g,
                    connectivity=True,
                    p_conn = {},
                    topology = {},
                    spatial_geometry={'id': None},
                    given_edges={}):
    """
    Build connections of a NG network providing connectivity dictionaries to pass to the network parameter structure.

    Input parameters:
    - N_e        : int   Number of E neurons
    - N_i        : int   Number of I neurons
    - N_g        : int   Number of G cells
    - connectivity : {True} | False : if False, return default N_conn,edges,coords
    - p_conn     : dict with any combination of the following keys ['ee','ie','ei','ii'] and values of associated connection probability
    - topology   : dict with keys ['ee','ie','ei','ii'] and values according to the connect() method
    - syn_to_ast : dict with keys ['ee','ie','ei','ii'] and values True | False
    - ast_to_syn : dict with keys ['ee','ie','ei','ii'] and values True | False

    Return:
    - N_conn : dict with keys ['e','i','g','ee','ie','ii','ei','gg','ng','gn','gee','gie','gei','gii'] and values the
             the associated number of connections
    - edges  : dict with keys ['ee','ie','ii','ei','gg','ng','gn'] and values the associated list of pairs of connections
    - coords : dict with keys ['e','i','g','ee','ie','ei','ii','id','dsize'] All entries except the last two contain coordinates of the different elements
    """

    # TODO: Does not handle weights on connections

    # Default connections pathways
    keys_conn = ['ee','ie','ii','ei','gg','gee','gie','gei','gii','eeg','ieg','eig','iig']
    # Default values
    network_topology = {}
    for k in keys_conn:
        if k in keys_conn[:4]:
            network_topology[k] = 'random-const-outdeg'
        elif k=='gg':
            network_topology[k] = None
        elif k in keys_conn[5:9]:
            network_topology[k] = 's2g-random-const-outdeg'
        else:
            network_topology[k] = 'fixed'
    # network_topology = {'random' for k in keys_conn if k in keys_conn[:5] else 's2g-random' if k in keys_conn[5:9] else 'fixed'}
    # network_topology = dict(list(zip(keys_conn,['random']*np.size(keys_conn)))) # All connections are randomly picked (except gliotransmitter ones)
    prob_conn = dict(list(zip(keys_conn,[1.]*np.size(keys_conn)))) # All connection probabilities are set to 1. initially

    # Custom parameters
    network_topology = gu.varargin(network_topology,**topology)
    prob_conn = gu.varargin(prob_conn,**p_conn)

    # Initialize number of connections
    N_conn = {'e': N_e, 'i': N_i, 'g': N_g}

    # Allocate cells
    # Spatial coordinates are provided as arrays of (N_cells x num_dimensions)
    coords = {}
    # The space of the network is estimated by the total number of synapses and the synaptic density

    if spatial_geometry['id']!=None:
        # Estimate number of synapses based on planar/spherical network
        # TODO: total_synapse_number should handle also the "clustered" case
        nsyn = total_synapse_number(N_e,N_i,prob_conn)
    # Estimations of network size (surface or volume) DO NOT take into account cell volume
    # NOTE: we temporarily modify spatial_geometry dictionary adding the characteristic size of the network
    # which is the side of the square in 'planar' geometry or the radius of the sphere in 'spherical' geometry

    # Only valid if spatial_geometry is a dictionary
    if spatial_geometry['id'] == 'planar':
        network_surface = nsyn/spatial_geometry['syn_density'] # This is assumed in um**2
        side = np.sqrt(network_surface)    # This is in um
        for k, n in N_conn.items():
            coords[k] = (np.c_[rpts_square(n, side=side)]).T
        # Update spatial_geometry
        spatial_geometry['dsize'] = side
    elif spatial_geometry['id'] == 'spherical':
        network_volume = nsyn/spatial_geometry['syn_density'] # This is assumed in um**3
        radius = (3*network_volume/(4*np.pi))**(1/3) # This is in um
        for k, n in N_conn.items():
            coords[k] = (np.c_[rpts_sphere(n, radius=radius)]).T
        # Update spatial_geometry
        spatial_geometry['dsize'] = radius
    else:
        # If spatial_geometry is None
        # This is the case of a non-spatial network (spatial_geometry=None)
        for k, n in N_conn.items():
            coords[k] = np.zeros((0, 3))
        spatial_geometry['dsize'] = None

    # Initialize connections
    keys = ['ee','ie','ei','ii','gg','gee','gie','gei','gii','eeg','ieg','eig','iig']
    edges = {}
    for k in keys:
        edges[k] = np.zeros((2,0))
        N_conn[k] = 0

    if connectivity:
        synaptic_clusters = {}
        # Neural connectivity (first check whether edges are given)
        if 'ee' in given_edges.keys():
            edges['ee'] = given_edges['ee']
        else:
            clusters = ['ee',spatial_geometry['e_cluster'],spatial_geometry['e_cluster']]
            edges['ee'],synaptic_clusters['ee'] = connect(N_e,N2=None,topology=network_topology['ee'],prob=prob_conn['ee'],
                                                          clusters=clusters,return_cluster_index=True)
        if 'ie' in given_edges.keys():
            edges['ie'] = given_edges['ie']
        else:
            clusters = ['ie',spatial_geometry['e_cluster'], spatial_geometry['i_cluster']]
            edges['ie'],synaptic_clusters['ie'] = connect(N_e,N2=N_i,topology=network_topology['ie'],prob=prob_conn['ie'],
                                                          clusters=clusters,return_cluster_index=True)
        if 'ei' in given_edges.keys():
            edges['ei'] = given_edges['ei']
        else:
            clusters = ['ei',spatial_geometry['i_cluster'], spatial_geometry['e_cluster']]
            edges['ei'],synaptic_clusters['ei'] = connect(N_i,N2=N_e,topology=network_topology['ei'],prob=prob_conn['ei'],
                                                          clusters=clusters,return_cluster_index=True)
        if 'ii' in given_edges.keys():
            edges['ii'] = given_edges['ii']
        else:
            clusters = ['ii',spatial_geometry['i_cluster'],spatial_geometry['i_cluster']]
            edges['ii'],synaptic_clusters['ii'] = connect(N_i,N2=None,topology=network_topology['ii'],prob=prob_conn['ii'],
                                                          clusters=clusters,return_cluster_index=True)

        for k in ['ee','ie','ei','ii']:
            N_conn[k] = np.shape(edges[k])[1]
            # Add synapse coordinates (commented: you run out of memory:: need to work on synaptic indexes on post-neurons)
            # coords[k] = synapse_coordinates(coords[k[-1]],coords[k[0]],edges[k],location='post')

        if N_g>0:
            # Connectivity of the glial network
            if 'gg' in given_edges.keys():
                edges['gg'] = given_edges['gg']
            else:
                if np.any(p_conn['gg']>0.):
                    # The N2 argument is actually not used at the moment in the connection of 'gg'
                    # synaptic clusters['gg'] are in practice gap junction connections // Spatial geometry in 'gg' is currently disabled (i.e. not passed)
                    clusters = ['gg',spatial_geometry['g_cluster'],spatial_geometry['g_cluster']]
                    edges['gg'] = connect(N_g,N2=None,topology=network_topology['gg'],prob=prob_conn['gg'],
                                                                  clusters=clusters,return_cluster_index=False)
                    # Make sure to select unique tuples
                    if np.size(edges['gg'])>0:
                        # The following is a jolly line to get unique unordered pairs from array i.e. (a,b) and (b,a) are treated the same since they will be doubled later
                        edges['gg'] = np.vstack(set(tuple(frozenset(gjc)) for gjc in set(list(zip(*edges['gg']))))).T
                else:
                    edges['gg'] = np.zeros((2,0))
            for k in ['ee','ie','ei','ii']:
                # Connections from synapses to astrocytes
                edges['g'+k] = np.zeros((2,0))
                # We use 'any' in the following rather than the raw p_conn>0 to check also for probability matrices
                if np.any(p_conn['g'+k]>0.) and (N_conn[k]>0):
                    if 'g'+k in given_edges.keys():
                        edges['g'+k] = given_edges['g'+k]
                    else:
                        # Automatically set coordinates of synapses to post -- this is necessary at the moment otherwise we run out of memory
                        clusters = ['g'+k,synaptic_clusters[k],spatial_geometry['g_cluster']]
                        edges['g'+k] = connect(N_conn[k],N2=N_g,topology=network_topology['g'+k],prob=prob_conn['g'+k],
                                                spatial_geometry=spatial_geometry,clusters=clusters,
                                                cds_glia=coords['g'],cds_neurons=coords[k[0]],syn_neu_indexes=edges[k][-1])
                N_conn['g'+k] = np.shape(edges['g'+k])[1]
                # Connections for gliotranmission (from astrocyte to synapses)
                edges[k+'g'] = np.zeros((2,0))
                # TODO: currently the glt connections are automatically implied as reversed (fixed) from incoming connections
                if np.any(p_conn[k+'g']>0):
                    if N_conn[k]>0:
                        if network_topology[k+'g']=='fixed':
                            # Currently only option: those synapses that excite glia are also affected by this latter
                            edges[k+'g'] = np.flip(edges['g'+k],axis=0)
                        # The following is a quick workaround to handle clusters without gliotransmission
                        if (len(spatial_geometry['g_cluster'])>1)and(np.size(p_conn[k+'g'])>1)and(np.count_nonzero(np.diff(p_conn[k+'g']))>0):
                            assert np.size(p_conn[k+'g'])==len(spatial_geometry['g_cluster']),'pconn['+k+'g] must be of size N_clusters when not a scalar'
                            to_remove = [np.nonzero(np.isin(edges[k+'g'][0],spatial_geometry['g_cluster'][i]))[0] for i,p in enumerate(p_conn[k+'g']) if p==0.0]
                            edges[k+'g'] = np.delete(edges[k+'g'],to_remove,axis=1)
                N_conn[k+'g'] = np.shape(edges[k+'g'])[1]

    # Embed relevant info on spatial geometry in coords
    coords['geometry'] = spatial_geometry['id']
    coords['dsize'] = spatial_geometry['dsize']

    return N_conn,edges,coords

#-----------------------------------------------------------------------------------------------------------------------
# UNIFORM SAMPLING ON DIFFERENT SURFACES
#-----------------------------------------------------------------------------------------------------------------------
@np.vectorize
def rpts_square(N_cells,side=1.,d_min=0.,totalPoints=None,show=False):
    """
    Provide random positions in a square of side 'side'.
    Allow to set minimal distance between points by d_min.

    N_cells : int   number of cells to position in the square
    side    : float {1} side of the square
    d_min   : float {0} minimum distance between points
    totalPoints : None | int>=N_cells  Number of points to look for positioning (None: default N_cells**2) (only relevand if d_min>0)
    show    : False | {True}  Show points (basic plotting)
    """

    N_cells = int(N_cells)

    if d_min==0:
        # There is no minimal distance for positioning the cells
        # Work on normalized data (keeperX and keeperY are the x,y normalized coordinates)
        # Must assure that all points are not coincident
        while True:
            keeperX = np.random.rand(N_cells)
            keeperY = np.random.rand(N_cells)
            if np.shape(np.unique(np.vstack((keeperX,keeperY)),axis=1))[-1]==N_cells:
                break
            else:
                print("Re-drawing for non-coincident points")
    else:
        # This case is by positioning cells at d_min from each other
        # Reason on normalized data coordinates
        minAllowableDistance = d_min/side
        assert minAllowableDistance<1,"minimal distance has to be < square side"
        numberOfPoints = N_cells

        # Create generic points
        if totalPoints==None: totalPoints = numberOfPoints**2
        totalPoints = int(totalPoints)
        x = np.random.rand(totalPoints)
        y = np.random.rand(totalPoints)

        # Initialize first point
        keeperY,keeperX = [],[]
        keeperX.append(x[0])
        keeperY.append(y[0])

        # Run over all points and make till it does not get N_cell points
        counter = 1
        for k in range(1,totalPoints):
            # Get trial point
            thisX = x[k]
            thisY = y[k]
            # Check distances between all other points
            distances = np.sqrt((thisX-np.asarray(keeperX))**2+(thisY-np.asarray(keeperY))**2)
            minDistance = np.amin(distances)
            if minDistance >= minAllowableDistance:
                keeperX.append(thisX)
                keeperY.append(thisY)
                counter += 1
            if counter >= numberOfPoints:
                break

        # Issue a warning in case was not able to position the specified number of cells
        if counter<numberOfPoints:
            print("WARNING: Position ",str(counter)," out of ",str(numberOfPoints)," requested cell. Try increasing totalPoints but computing time could extend.")

        # Convert to array
        keeperX = np.atleast_1d(keeperX)
        keeperY = np.atleast_1d(keeperY)

    if show:
        plt.plot(side*keeperX, side*keeperY, 'kx')
        plt.show()

    return side*keeperX,side*keeperY

@np.vectorize
def rpts_sphere(N_pts,radius=1.):
    """
    Generate random points on a sphere (from uniform distribution), according to the random algorithm in:
    https://www.cs.cmu.edu/~mws/rpos.html.

    Input arguments:
    - N_pts     : Number of points to generate
    - radius    : sphere radius

    Return:
    - x,y,z     : cartesian coordinates for the points.

    Maurizio De Pitta', The University of Chicago, Chicago, May 3, 2016.
    """

    # Pre-process empty data-sets
    N_pts = max(N_pts,0)

    radius = float(np.abs(radius))
    z = np.random.uniform(-radius,radius,N_pts)
    phi = np.random.uniform(0.,2*np.pi,N_pts)
    theta = np.arcsin(z/radius)
    # Convert to Cartesian coordinates
    x = radius*np.cos(theta)*np.cos(phi)
    y = radius*np.cos(theta)*np.sin(phi)
    return  x,y,z

def rpts_hemisphere(N_pts,radius=1.):
    # Pre-process empty data-sets
    N_pts = max(N_pts,0)

    z = np.random.uniform(0.,radius,N_pts)
    r = np.sqrt(radius**2.-z**2)
    phi = np.random.uniform(0.,2*np.pi,N_pts)

    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return x,y,z

def provide_lat_long(x,y,z,radius=1.):
    lat = np.arcsin(z/radius)
    lon = np.arctan2(y,x)
    return lat,lon

def great_circle_distance(p0,p1,radius=1.):
    # Compute distance between two points on a sphere (or hemisphere) using cartesian coordinates
    # First return latitudes and longitudes of the two points
    l0, phi0 = provide_lat_long(*zip(p0), radius=radius)
    l1, phi1 = provide_lat_long(*zip(p1), radius=radius)
    dl = l0-l1
    # Implement Vincenty Inverse Formula on the sphere (from https://en.wikipedia.org/wiki/Great-circle_distance)
    N = np.hypot(np.cos(phi1)*np.sin(dl),np.cos(phi0)*np.sin(phi1)-np.sin(phi0)*np.cos(phi1)*np.cos(dl))
    D = np.sin(phi0)*np.sin(phi1) + np.cos(phi0)*np.cos(phi1)*np.cos(dl)
    return radius*np.arctan2(N,D)

def cell_domains(cell_pts,geometry='planar',size=1.,show=False):
    # Points must be provided as tuple of ndims tuples each of (1 x npts) per dimension (typically the output of rpts_* methods
    # Compute glial domains by Voronoi tessellation / WARNING: Uses different methods for 2d vs. 3d
    # The show method also reflect different methods used to plot the results
    if geometry=='planar':
        box = freud.box.Box.square(L=size)
        voro = freud.locality.Voronoi()
        # make cell_pts in 3d as required by freud.locality.Voronoi
        pts = np.vstack((cell_pts,np.zeros((1,np.size(cell_pts)//2)))).T
        domains = voro.compute((box,pts)).polytopes
        # domains = space.Voronoi(glia_pts)
        if show:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # TODO: needs adjustment : The tessellation is fine though but out of the box
            voro.plot(ax,color_by_sides=False)  # This is part of the freud.locality.Voronoi class
            ax.scatter(xy[0], xy[1], s=10, c='k')
    elif geometry=='spherical':
        # In this case size is the radius of the sphere
        domains = space.SphericalVoronoi(cell_pts,radius=size,center=(0,0,0))
        domains.sort_vertices_of_regions()
        if show:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for region in domains.regions:
                random_color = mplc.rgb2hex(np.random.rand(3))
                polygon = art3d.Poly3DCollection([domains.vertices[region]], alpha=1.0)
                polygon.set_color(random_color)
                ax.add_collection3d(polygon)
    return domains

def find_cellular_domains(glia_pts,neurons_pts,geometry='planar',radius=None):
    # Points must be provided as an array of [npoints x 3]
    # The (c)KDTree is assumed to work also in the spherical case when using p=2 in the query (i.e. Euclidean distance)
    if geometry=='planar':
        metric = nbors.DistanceMetric.get_metric('euclidean')
        gp_ = glia_pts
        np_ = neurons_pts
    elif geometry=='spherical':
        assert radius>0.,"Sphere radius must be specified and >0"
        metric = nbors.DistanceMetric.get_metric('haversine')
        # This metric also requires the points to be given in lat,long
        gp_ = provide_lat_long(glia_pts[:,0],glia_pts[:,1],glia_pts[:,2],radius=radius)
        np_ = provide_lat_long(neurons_pts[:,0],neurons_pts[:,1],neurons_pts[:,2],radius=radius)
    celldom = nbors.KDTree(gp_,metric=metric)
    domain_index = celldom.query(np_,k=1,return_distance=False,dualtree=True)
    return np.concatenate(domain_index)

# def g_phi_pdf(N_pts,R,r):
#     #Rejection sampler
#     xvec = np.random.uniform(0,2*np.pi,N_pts)
#     yvec = np.random.uniform(0,1./np.pi,N_pts)
#     fx = (1+(r/R)*np.cos(xvec))/(2*np.pi)
#     return xvec[yvec<fx]

# def rpts_torus(N_pts,R=1.,r=0.3):
#     """
#     Uniformly distributed points on a torus.
#
#     Algorithm based on Diaconis, Holmes and Shahshahani, Sampling from a Manifold, Advances Modern Statistical Theory
#     and Applications, 2013, pp. 102--125.
#     http://statweb.stanford.edu/~cgates/PERSI/papers/sampling11.pdf
#
#     Args:
#         N_pts:
#         R:
#         r:
#
#     Returns:
#
#     Maurizio De Pitta', The University of Chicago, Chicago, May 3, 2016.
#     """
#
#     # Pre-process empty data-sets
#     N_pts = max(N_pts,0)
#
#     assert R>=r, "R (torus radius) must be >= r (inner radius)"
#
#     phi = np.random.uniform(0.,2*np.pi,N_pts)
#     theta = np.array([])
#     while theta.size<N_pts: theta = np.r_[theta,g_phi_pdf(N_pts,R,r)]
#     theta = theta[:N_pts]
#
#     # Convert to Cartesian coordinates
#     x = (R+r*np.cos(theta))*np.cos(phi)
#     y = (R+r*np.cos(theta))*np.sin(phi)
#     z = r*np.sin(theta)
#
#     return x,y,z
#
# def rpts_disk(N_pts,radius=1.,z_coord=0.):
#     """
#     Generate random points on a disk (from uniform distribution).
#
#     Algorithm:
#     You can think about this solution as follows. If you took the circle, cut it, then straightened it out, you'd get a
#     right-angled triangle. Scale that triangle down, and you'd have a triangle from (0, 0) to (1, 0) to (1, 1) and back
#     again to (0, 0). All of these transformations change the density uniformly. What you've done is uniformly picked a
#     random point in the triangle and reversed the process to get a point in the circle.
#
#     Input arguments:
#     - N_pts     : Number of points to generate
#     - radius    : disk radius
#
#     Return:
#     - x,y,z     : cartesian coordinates for the points.
#
#     Maurizio De Pitta', The University of Chicago, Chicago, May 3, 2016.
#     """
#     # Pre-process empty data-sets
#     N_pts = max(N_pts,0)
#
#     phi = np.random.uniform(0.,2*np.pi,N_pts)
#     r = np.random.uniform(-radius,radius,N_pts)
#     x = r*np.cos(phi)
#     y = r*np.sin(phi)
#     return x,y,z_coord*np.ones(N_pts)
#
# def rpts_annulus(N_pts,radius_min=0.5,radius_max=1.0,z_coord=0.):
#     """
#     Generate random points uniformly distributed on an annulus.
#
#     Algorithm:
#     - Generate theta uniformly.
#     - Generate r randomly from power law "r^1".
#     - Convert to Cartesian coordinates.
#
#     http://stackoverflow.com/questions/9048095/create-random-number-within-an-annulus
#
#     Input arguments:
#     - N_pts     : Number of points to generate
#     - radius_min: inner annulus radius
#     - radius_max: outer annulus radius
#
#     Return:
#     - x,y     : cartesian coordinates for the points.
#
#     Maurizio De Pitta', The University of Chicago, Chicago, May 3, 2016.
#     """
#     # Pre-process empty data-sets
#     N_pts = max(N_pts,0)
#
#     # Power-law generator
#     r_power_law = lambda rmin,rmax,n : ((rmax**(n+1) - rmin**(n+1))**np.random.uniform(size=N_pts) + rmin**(n+1))**(1/(n+1))
#
#     # Build pointsw
#     theta = np.random.uniform(0.,2*np.pi,N_pts)
#     r = r_power_law(radius_min,radius_max,1.)
#     x = r*np.cos(theta)
#     y = r*np.sin(theta)
#     return x,y,z_coord*np.ones(N_pts)
#
# #-----------------------------------------------------------------------------------------------------------------------
# # POSITION IN THE PHYSICAL SPACE
# #-----------------------------------------------------------------------------------------------------------------------
# def translate3d(PTS,x_shift=0.,y_shift=0,z_shift=0.):
#     return PTS+np.tile(np.array([x_shift,y_shift,z_shift]),(np.shape(PTS)[0],1))
#
# def rotate3d(PTS,x_angle=0.,y_angle=0.,z_angle=0.):
#     # Pre-processing of empty data sets
#     if np.size(PTS)==0: return np.zeros((0,3))
#     # Rotation matrices
#     R_x = np.array([[1,0,0],[0,np.cos(x_angle),-np.sin(x_angle)],[0,np.sin(x_angle),np.cos(x_angle)]])
#     R_y = np.array([[np.cos(y_angle),0,np.sin(y_angle)],[0,1,0],[-np.sin(y_angle),0,np.cos(y_angle)]])
#     R_z = np.array([[np.cos(z_angle),-np.sin(z_angle),0],[np.sin(z_angle),np.cos(z_angle),0],[0,0,1]])
#     R = np.dot(np.dot(R_z,R_y),R_x)
#     return np.dot(PTS,R.T)
#
# def project_on_sphere(p,radius=1.):
#     """
#     Project point p(x,y,z) to a sphere of radius r (centered in the origin)
#     """
#
#     # Convert to spherical coordinates
#     x,y,z = tuple(p)
#     r = np.sqrt(x**2+y**2+z**2)
#     theta = np.arcsin(z/r)
#     phi = np.arctan(y/x)
#     # Re-convert to cartesian coordinates but on the sphere's surface
#     xs = radius*np.cos(theta)*np.cos(phi)
#     ys = radius*np.cos(theta)*np.sin(phi)
#     zs = radius*np.cos(theta)
#     return xs,ys,zs
#
# def pts_synapses(P1,P2,edges,sfactor=0.75,x_shift=0,y_shift=0):
#     # Pre-processing empty data sets
#     if np.size(P1)*np.size(P2)==0: return np.zeros((0,np.shape(P1)[1]))
#     # Check that shifts are integers as edges are between ranked data points
#     assert type(x_shift)==int, "x_shift for indexing must be integer"
#     assert type(y_shift)==int, "y_shift for indexing must be integer"
#     pre = P1[edges[:,0]+x_shift]
#     post = P2[edges[:,1]+y_shift]
#     return pre+(post-pre)*sfactor
#
# def nodes_position(N_e,N_i,N_g,edges_ee,edges_ei,edges_ie,edges_ii,
#                    syn_dist=0.1,layout='sphere-torus'):
#     """
#
#     :param N_e:
#     :param N_i:
#     :param N_g:
#     :param edges_ee:
#     :param edges_ei:
#     :param edges_ie:
#     :param edges_ii:
#     :param syn_dist:    Distance of synapse from postsynaptic node
#     :param configuration:
#     :return:
#     """
#
#
#     # Retrieve "unsplit" XY connections (i.e. original EE,IE... etc)
#     direct_edges = lambda edgs,N : np.asarray([edgs[:N,0],edgs[N:2*N,1]]).T
#
#     if (layout in ['sphere-torus','default']):
#         # For now only in 3D
#         pts_e = rotate3d(list(zip(*rpts_hemisphere(N_e,radius=1.))),x_angle=-np.pi/2)
#         pts_i = rotate3d(list(zip(*rpts_hemisphere(N_i,radius=1.))),x_angle=np.pi/2)
#         pts_a = np.asarray(list(zip(*rpts_torus(N_g,R=1.7,r=0.4))))
#     elif layout=='all-to-one':
#         # Applies only to EE populations with one astrocyte
#         pts_e = rotate3d(list(zip(*rpts_disk(N_e-1,radius=1.))),y_angle=-np.pi/2)
#         if (N_e>0): pts_e = np.vstack((pts_e,np.array([0,0,1])))
#         pts_i = rotate3d(list(zip(*rpts_disk(N_i-1,radius=1.))),x_angle=np.pi/2)
#         if (N_i>0): pts_i = np.vstack((pts_i,np.array([0,0,1])))
#         pts_a = np.asarray([0,0,-1])
#
#     pts_ee = pts_synapses(pts_e,pts_e,edges_ee,sfactor=syn_dist)
#     pts_ie = pts_synapses(pts_e,pts_i,edges_ie,y_shift=-N_e,sfactor=syn_dist)
#     pts_ei = pts_synapses(pts_i,pts_e,edges_ei,x_shift=-N_e,sfactor=syn_dist)
#     pts_ii = pts_synapses(pts_i,pts_e,edges_ii,x_shift=-N_e,y_shift=-N_e,sfactor=syn_dist)
#
#     # Stack points in an orderly fashion and return nodes_position dictionary
#     pts = np.vstack((pts_e,pts_i,pts_ee,pts_ie,pts_ei,pts_ii,pts_a))
#     return pts
#
# #-----------------------------------------------------------------------------------------------------------------------
# # NETWORK CONNECTION VISUALIZATION
# #-----------------------------------------------------------------------------------------------------------------------
# def conn_split(edge_list,idx_shift=0):
#     # if ((dest_shift==0) and (source_shift!=dest_shift)): dest_shift = source_shift
#     N_s = np.shape(edge_list)[0]
#     # Edge list From source neurons to synapses
#     e_1s = np.vstack((edge_list[:,0],idx_shift+np.arange(N_s))).T
#     # Edge list from synapses to target neurons
#     e_2s = np.vstack((idx_shift+np.arange(N_s),edge_list[:,1])).T
#     return N_s,e_1s,e_2s
#
# def edges_color(edges,N_e,N_i,N_g,N_ee,N_ei,N_ie,N_ii):
#     colors = {'ee' : 'r',
#               'ie' : 'm',
#               'ei' : 'c',
#               'ii' : 'b',
#               'gg' : 'g',
#               'as' : 'y'}
#     ec = np.array([colors['ee']]*np.shape(edges)[0]) # convert to np.array for easy indexing
#     aux = list(zip(*edges))
#     key_lims = ['ie','ei','ii','gg']
#     idx_lims = dict(list(zip(key_lims,N_e+N_i+np.array([N_ee,N_ee+N_ei,N_ee+N_ei+N_ie,N_ee+N_ei+N_ie+N_ii]))))
#     # for _,k in enumerate(key_lims[:-3]):
#     for i,k in enumerate(key_lims[:-1]):
#         index = ((aux[0]>=idx_lims[k])&(aux[0]<idx_lims[key_lims[i+1]]))|((aux[1]>=idx_lims[k])&(aux[1]<idx_lims[key_lims[i+1]]))
#         ec[index] = colors[k]
#     # A-->S connections (gliotransmission)
#     ec[(aux[0]>=idx_lims['gg'])] = colors['as']
#     # gg connections
#     ec[(aux[0]>=idx_lims['gg'])&(aux[1]>=idx_lims['gg'])] = colors['gg']
#     return ec
#
# def show_network(C_ee,C_ie,C_ii,C_ei,C_gg,C_ng,C_gn=None, layout='default', syn_dist=0.1):
#     # TODO: C_gn is currently not implemented. It just assumes that all excitatory synapses are modulated
#     colors = {'e' : 'r',
#               'i' : 'b',
#               'ee' : 'r',
#               'ie' : 'm',
#               'ii' : 'b',
#               'ei' : 'c',
#               'exc': 'm',
#               'inh': 'c',
#               'a' : 'g'}
#     nsize = {'e' : 100,
#              'i' : 100,
#              'exc': 10,
#              'inh': 10,
#              # 'exc': 300,
#              # 'inh': 300,
#              'a' : 200}
#
#     """
#     The Connectivity matrix of the whole network is so composed:
#     - xx denote connectivity of neural populations needed originally for correct indexing
#     - the (*) denote individual (derived) connectivity matrices
#
#         E   I   EE  IE  EI  II  A
#     E   ee  ie  *   *
#     I   ei  ii          *   *
#     EE  *                       *
#     IE      *                   *
#     EI  *                       *
#     II      *                   *
#     A           *   *           *
#
#     """
#
#     # Complete graph
#     N_e = np.shape(C_ee)[0]
#     N_i = np.shape(C_ii)[0]
#     N_g = np.shape(C_gg)[0]
#
#     # Generate Graph object
#     ngn = nx.DiGraph()
#
#     # Compute edges for E,I interconnections
#     edges_ee = edges_list(C_ee)
#     edges_ie = edges_list(C_ie,y_shift=N_e)
#     edges_ei = edges_list(C_ei,x_shift=N_e)
#     edges_ii = edges_list(C_ii,x_shift=N_e,y_shift=N_e)
#     # Resolve synapses as individual nodes
#     N_ee,edges_ees,edges_see = conn_split(edges_ee,idx_shift=N_e+N_i)
#     N_ie,edges_ies,edges_sie = conn_split(edges_ie,idx_shift=N_e+N_i+N_ee)
#     N_ei,edges_eis,edges_sei = conn_split(edges_ei,idx_shift=N_e+N_i+N_ee+N_ie)
#     N_ii,edges_iis,edges_sii = conn_split(edges_ii,idx_shift=N_e+N_i+N_ee+N_ie+N_ei)
#     # Glia network
#     edges_gg = edges_list(C_gg,x_shift=N_e+N_i+N_ee+N_ie+N_ei+N_ii,y_shift=N_e+N_i+N_ee+N_ie+N_ei+N_ii)
#     # Synapse-->Astrocyte connections
#     edges_ng = edges_list(C_ng,x_shift=N_e+N_i,y_shift=N_e+N_i+N_ee+N_ie+N_ei+N_ii)
#     # Astrocyte-->Synapse connections (only on excitatory synapses)
#     edges_gn = np.fliplr(edges_ng[:N_ee+N_ie])
#
#     # Add all nodes
#     ngn.add_nodes_from(np.arange(N_e+N_i+N_ee+N_ie+N_ei+N_ii+N_g))
#     node_color = [colors['e']]*N_e + [colors['i']]*N_i + [colors['exc']]*(N_ee+N_ie) + [colors['inh']]*(N_ei+N_ii) + [colors['a']]*N_g
#     node_size = [nsize['e']]*N_e + [nsize['i']]*N_i + [nsize['exc']]*(N_ee+N_ie) + [nsize['inh']]*(N_ei+N_ii) + [nsize['a']]*N_g
#
#     # Add edges
#     ngn.add_edges_from(np.r_[edges_ees,edges_see]) # Add EE
#     ngn.add_edges_from(np.r_[edges_ies,edges_sie]) # Add IE
#     ngn.add_edges_from(np.r_[edges_eis,edges_sei]) # Add EI
#     ngn.add_edges_from(np.r_[edges_iis,edges_sii]) # Add II
#     ngn.add_edges_from(edges_gg) # Add GG (glia network)
#     ngn.add_edges_from(edges_ng) # Add S->A (synapse-to-astro connections)
#     ngn.add_edges_from(edges_gn) # Add A->S (gliotransmitter connections)
#
#     # svu.savedata([ngn,N_e,N_i,N_g,N_ee,N_ei,N_ie,N_ii,node_color,node_size],'temp_ngn.pkl')
#     # [ngn,N_e,N_i,N_g,N_ee,N_ei,N_ie,N_ii,node_color,node_size] = svu.loaddata('temp_ngn.pkl')
#
#     # Retrieve colors for edges
#     node_pos = nodes_position(N_e,N_i,N_g,edges_ee,edges_ei,edges_ie,edges_ii,layout=layout,syn_dist=syn_dist)
#     # node_pos = np.vstack(nx.spring_layout(ngn,dim=3).values())
#     edge_color = edges_color(nx.edges(ngn),N_e,N_i,N_g,N_ee,N_ei,N_ie,N_ii)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Plot Nodes
#     node_collection = ax.scatter(node_pos[:, 0], node_pos[:, 1], node_pos[:,2],
#                                  s=node_size,
#                                  c=node_color,
#                                  # marker=node_shape,
#                                  # cmap=cmap,
#                                  # vmin=vmin,
#                                  # vmax=vmax,
#                                  # alpha=alpha,
#                                  # linewidths=linewidths,
#                                  # label=label
#                                  )
#     node_collection.set_zorder(2)
#
#     # Plot edges
#     edge_pos = np.asarray([(node_pos[e[0]], node_pos[e[1]]) for e in nx.edges(ngn)])
#     edge_collection = plt3d.art3d.Line3DCollection(edge_pos,
#                                      colors=edge_color,
#                                      # linewidths=lw,
#                                      antialiaseds=(1,),
#                                      # linestyle=style,
#                                      transOffset = ax.transData,
#                                      )
#
#     edge_collection.set_zorder(1)  # edges go behind nodes
#     ax.add_collection(edge_collection)
#     plt.show()
#
# def show_connectivity_matrix(Nconn,edges,ax=None,show_blocks=False,**kwargs):
#
#     # TODO: Include colorbar for G cell number
#
#     # Plotting defaults
#     pars = {'nstep': 25,
#             'spine_position': 6.0,
#             'sf' : 5.,  # Scaling factor for scatter plot
#             'lw' : 5,
#             'alw': 1.5,
#             'afs': 10,
#             'lfs': 14,
#             'offset': 1., # Offset for x-y rectangle lines (outside/inside +/- axis limits)
#             'ecolor': pu.palette('Red'),
#             'icolor': pu.palette('Blue'),
#             'gcolor': pu.palette('LightGray'),
#             'cmap': 'viridis'}
#     pars = gu.varargin(pars,**kwargs)
#
#     # Retrieve some useful number
#     N_e = Nconn['e']
#     N_i = Nconn['i']
#     N_g = Nconn['g']
#
#     if ax==None:
#         if not show_blocks:
#             margins = [0.13,0.97,0.12,0.97]
#             fig,ax = plt.subplots(nrows=1, ncols=1,
#                 gridspec_kw={'left': margins[0], 'right': margins[1],
#                              'bottom': margins[2], 'top': margins[3]},
#                 figsize=(6.,6.))
#             # For easiness of handling with respect to the colorbar case, we convert the scalar ax into a (ax) tuple
#             ax = np.atleast_1d(ax)
#         else:
#             space = 0.05
#             margins = [0.1, 0.97, 0.12, 0.97]
#             fig, ax = plt.subplots(nrows=2, ncols=2,
#                 # sharex=True,sharey=True,
#                 gridspec_kw={'left': margins[0], 'right': margins[1],
#                              'bottom': margins[2], 'top': margins[3],
#                              'height_ratios': [1., N_i/N_e],
#                              'width_ratios': [1., N_i/N_e],
#                              'hspace': space,
#                              'wspace': space},
#                 figsize=(8., 8.))
#             # For easiness of handling with respect to the colorbar case, we convert the scalar ax into a (ax) tuple
#             ax = ax.ravel()
#
#     # Create a discrete colormap
#     vlims = np.asarray([0,N_g-1])
#     cmap = pu.discrete_cmap(N_g-1,pars['cmap'])
#     cmap.set_under('red')
#
#     if not show_blocks:
#         # Handling of not blocks is completely different from blocks
#         # Generate a matrix ExI
#         EIG = -2*np.ones((N_e+N_i,N_e+N_i),dtype=float)
#         # EIG = np.empty((N_e + N_i, N_e + N_i)).fill(np.nan)
#         for k in ['ee','ei','ie','ii']:
#             ns = 0 if k[1]=='e' else N_e
#             nt = 0 if k[0]=='e' else N_e
#             if np.size(edges['g'+k])>0:
#                 EIG[ns+edges[k][0],nt+edges[k][-1]] = edges['g'+k][-1]
#             else:
#                 EIG[ns+edges[k][0],nt+edges[k][-1]] = -1
#
#         # Convert EIG to masked array
#         EIG = ma.masked_where(EIG<-1,EIG)
#         ax[0].imshow(EIG,cmap=cmap,vmin=vlims[0],vmax=vlims[-1],aspect='equal',interpolation=None)
#     else:
#         # In this case we plot directly as scatter plot, the individual edge points
#         xlims = [[0,N_e],[0,N_i],[0,N_e],[0,N_i]] # Used by rectangle
#         ylims = [[0,N_e],[0,N_e],[0,N_i],[0,N_i]] # Used by rectangle
#         xticks = [[],[],np.arange(0,N_e+1,(N_e+N_i)//pars['nstep']),np.arange(0,N_i+1,(N_e+N_i)//pars['nstep'])]
#         yticks = [np.arange(0, N_e+1, (N_e + N_i)//pars['nstep']), [], np.arange(0, N_i+1, (N_e + N_i)//pars['nstep']), []]
#         xticklabels = [[str(v+(v==0)*1) for v in xt] for xt in xticks]
#         yticklabels = [[str(v + (v==0)*1) for v in yt] for yt in yticks]
#         spineset = [['left'],[],['left','bottom'],['bottom']]
#         ecl = [pars['ecolor'],pars['ecolor'],pars['icolor'],pars['icolor']]
#         fill = [True,True,False,False]
#         fc = [pars['gcolor'],pars['gcolor'],None,None]
#         xlbl = ['','','Post E','Post I']
#         ylbl = ['Pre E','','Pre I','']
#
#         # Populate individual matrices
#         for i,k in enumerate(['ee','ie','ei','ii']):
#             # Add boxes for synapse types
#             ax[i].add_patch(mpatches.Rectangle((xlims[i][0]-pars['offset'], ylims[i][0]-pars['offset']),
#                 xlims[i][-1]+pars['offset'],  # width
#                 ylims[i][-1]+pars['offset'],  # height
#                 fill=False,
#                 edgecolor=ecl[i],
#                 linewidth=pars['lw'],
#                 linestyle='-'))
#             if np.size(edges['g'+k])>0:
#                 # Edges are reverted to match the ij matrix visualization standard
#                 ax[i].scatter(edges[k][-1],edges[k][0], s=pars['sf'], c=edges['g' + k][-1], marker='s',
#                     cmap=pars['cmap'], vmin=vlims[0],vmax=vlims[-1],
#                     zorder=10)
#             else:
#                 ax[i].scatter(edges[k][0],edges[k][-1], s=pars['sf'], c = -1, marker='s',
#                     cmap=pars['cmap'], vmin=vlims[0], vmax=vlims[-1],
#                     zorder=10)
#
#             if fill[i]:
#                 # Add color for gliotransmission (currently fixed to EE and IE)
#                 ax[i].set_facecolor(fc[i])
#
#             # # Adjust axes
#             ax[i].set_clip_on(False)
#             ax[i].autoscale(enable=True,axis='both',tight=True)
#             ax[i].invert_yaxis()
#             ax[i].set(xticks=xticks[i],xticklabels=xticklabels[i],
#                 yticks=yticks[i],yticklabels=yticklabels[i])
#             pu.adjust_spines(ax[i],spineset[i],position=pars['spine_position'])
#             if i!=1:
#                 pu.set_axlw(ax[i], lw=pars['alw'])
#                 pu.set_axfs(ax[i], fs=pars['afs'])
#
#             # Add label
#             if i in [2,3]:
#                 ax[i].set_xlabel(xlbl[i],fontsize=pars['lfs'])
#             if i in [0,2]:
#                 ax[i].set_ylabel(ylbl[i],fontsize=pars['lfs'])
#
#             # Add colorbar (with G cell number)
#             # cb = plt.colorbar()
#
#     return ax
#
# #-----------------------------------------------------------------------------------------------------------------------
# # TESTING ROUTINES
# #-----------------------------------------------------------------------------------------------------------------------
# def visualize_ngn(N=10, layout='default'):
#
#     if layout=='default':
#         # sphere-torus network
#         N_e = N
#         N_i = N
#         N_g = N
#         C = 2.
#         A = 2.
#         # C_ee,C_ie,C_ii,C_ei,C_gg,C_ng = ngn_connections(N_e,N_i,N_g,p_ee=C/N_e,p_ei=C/N_i,p_ii=C/N_i,p_ie=C/N_e,p_a=A/N_g)
#         p_conn = dict(list(zip(['ee','ie','ii','ei','gg'],[C/N_e,C/N_e,C/N_i,C/N_i,A/N_g])))
#         C_conn,N_conn,_ = ngn_connections(N_e,N_i,N_g,p_conn=p_conn)
#         show_network(C_conn['ee'],C_conn['ie'],C_conn['ii'],C_conn['ei'],C_conn['gg'],C_conn['ng'],C_conn['gn'])
#     elif layout=='all-to-one':
#         N_e = N
#         N_i = 0
#         N_g = 1
#         pars_lif,pars_mhv,pars_glt,pars_prer,pars_conn,C_conn,N_conn,edges,_ = ngn_parameters(N_e+1,N_i,N_g,
#                                                                       ngn_setup='single-astro',
#                                                                       topology={},syn_to_ast={},ast_to_syn={},p_conn={},
#                                                                       ne_pars={},ni_pars='same',ng_pars={},
#                                                                       ee_pars='default',ie_pars='same',ii_pars='default',ei_pars='same',
#                                                                       glt_ee_pars='default',glt_ie_pars='same',glt_ii_pars='default',glt_ei_pars='same')
#         show_network(C_conn['ee'],C_conn['ie'],C_conn['ii'],C_conn['ei'],C_conn['gg'],C_conn['ng'],layout='all-to-one',syn_dist=0.9)

if __name__ == "__main__":
    #-------------------------------------------------------------------------------------------------
    # Testing connectivity
    #-------------------------------------------------------------------------------------------------
    # N1 = 1e6
    # arg = 0.3
    # arg = 100
    # # N2 = None
    # N2 = 5000
    # topology = 's2a-random-fixed-connectivity'
    # e = connect(N1, N2=N2, topology=topology, arg=arg, seed=None)
    # # for i in range(np.shape(e)[1]):
    # #     print(e.T[i])
    # # visualize_ngn(N=10,layout='default')

    # # Testing connections
    # N_e = 4e3
    # N_i = 1e3
    # N_g = 2.5e3
    # p = 0.1
    # p_conn = {'ee': p, 'ie': p, 'ei': p, 'ii': p, 'gg': 0.4}
    # N,e = ngn_connections(N_e,N_i,N_g,p_conn=p_conn)
    # print(N)
    # print(e)

    #-------------------------------------------------------------------------------------------------
    # Testing spatial network
    #-------------------------------------------------------------------------------------------------
    # npts = 4000
    # xn,yn,zn = rpts_hemisphere(npts, radius=1.)
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # #
    # # ax.scatter(x,y,z)
    #
    # gpts = 2500
    # xg,yg,zg = rpts_hemisphere(gpts, radius=1.)
    # dom = cellular_domains(np.c_[xg,yg,zg],np.c_[xn,yn,zn])
    # print(dom)
    # print(np.amin(dom),np.amax(dom))
    # _,c = np.unique(dom,return_counts=True)
    # print(np.mean(c))

    #-------------------------------------------------------------------------------------------------
    # Testing Voronoi tessellation
    #-------------------------------------------------------------------------------------------------
    N = 100
    xy = rpts_square(N,side=1.,show=False)
    cell_domains(xy,geometry='planar',size=1.,show=True)

    plt.show()