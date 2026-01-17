# Custom imports
from brian2 import *
from brian2.units import *
from brian2.units.constants import faraday_constant as F
from functions_brianlib import ThermalPotential,NernstPotential,Hill,IversonBrackets

#-----------------------------------------------------------------------------------------------------------------------
# CPP Settings
#-----------------------------------------------------------------------------------------------------------------------
code_dir = './codegen'
prefs.GSL.directory = '/usr/include/'  ## The directory where the GSL library headings are found
set_device('cpp_standalone',directory=code_dir,build_on_run=False)
prefs.devices.cpp_standalone.openmp_threads = 2  ## The number of threads used in the parallelization (machine-dependent)
prefs.logging.file_log = False
prefs.logging.delete_log_on_exit = True

def synaptic_connection(stim_source, snc_target, params, name='syn*', dt=0.1*us, delay=None):
    eqs = Equations('''
        # Bound fraction of postsynaptic receptors (assuming no activation at Nt_s=0) 
        J_rec = -r_clip/tau_r+J*(1-r_clip)*Nt_s_post : 1/second
        J_r_post = J_rec : 1/second (summed)
        dr/dt = J_rec   : 1 (clock-driven)
        r_clip = clip(r,0,1)        : 1        
        ''')
    on_pre = '''
             n_s_post += Nt_rel*Lambda_s
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

def periodic_nodes(N,rates,name='period*',dt=None):
    eqs = Equations('''
                    rate : Hz
                    dv/dt = rate : 1
                    ''')
    cells = NeuronGroup(N,eqs,
        threshold='v>=1.0',
        reset='v=0.0',
        name=name,
        method='euler',
        dt=dt)
    cells.v = 0.0
    cells.rate = rates
    return cells

def ecs_target(params,name='ecs',dt=0.1*us):
    eqs = Equations('''
            J_r : 1/second
            J_diff = -D*(n_s/Lambda_s-n_e/Lambda_e) : mmolar/second
            Nt_e = clip(G0_e+n_e/Lambda_e, 0*mmolar, inf*mmolar)        : mmolar
            Nt_s = clip(G0_e+n_s/Lambda_s, 0*mmolar, inf*mmolar)        : mmolar            
            dn_e/dt = -Lambda_s*J_diff - Lambda_e*D*(Nt_e-G0_e) : mole
            dn_s/dt = J_diff*Lambda_s - J_r*R_T*Lambda_s : mole
          ''')

    ecs = NeuronGroup(1, eqs,
                      namespace=params,
                      name=name,
                      method='rk4',
                      dt=dt,
                      order=10)

    ecs.n_e = 0*mole
    ecs.n_s = 0*mole

    return ecs

#-----------------------------------------------------------------------------------------------------------------------
# Parameter generator
#-----------------------------------------------------------------------------------------------------------------------
def ecs_parameters(**kwargs):
    pars = {# Extracellular concentrations
            'G0_e'    : 25*nmolar,
            'Lambda_e' : 500*um**3,
            'Lambda_s' : 8.75e-3*um**3,
            'D' : 10/second,
            'tau_r' : 20*ms,
            'Nt_rel': 10*umolar,
            'J' : 1/umolar/second,
            'R_T': 10*umolar
    }
    return pars

def ecs_simulate():
    device.delete(force=True)

    # Generate Model
    source = periodic_nodes(1,10*Hz,dt=1*ms)
    pars_ecs = ecs_parameters()
    ecs = ecs_target(pars_ecs,name='ecs*',dt=0.1*us)
    syn = synaptic_connection(source, ecs, pars_ecs, name='syn*', dt=0.1*us, delay=None)
    i_pre = np.atleast_1d([0]).astype(int)
    j_pst = np.atleast_1d([0]).astype(int)
    syn.connect(i=i_pre,j=j_pst)

    # Monitors
    mon_spk = SpikeMonitor(source, record=True, name='spk')
    mon = StateMonitor(ecs,variables=['Nt_e','Nt_s'],record=True,dt=0.5*ms,name='mon')

    # Network Simulation
    network = Network([source,ecs,syn,mon_spk,mon])
    network.run(duration=1*second,report='text')
    device.build(directory=code_dir,clean=True)

    _,axs = plt.subplots(3, 1, figsize=(6, 10))
    axs[0].vlines(mon_spk.t_, mon_spk.i[:].T, mon_spk.i[:].T+0.9)
    axs[1].plot(mon.t_,mon.Nt_e[:].T,'b-')
    axs[2].plot(mon.t_, mon.Nt_s[:].T, 'm-')

#-----------------------------------------------------------------------------------------------------------------------
# Running Tests
#-----------------------------------------------------------------------------------------------------------------------
if __name__=="__main__":
    ecs_simulate()

    plt.show()