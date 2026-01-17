from brian2 import *

# Parameters
S_N = 20000*umetre**2
Om_N = 5000*umetre**3
C_N = (1*ufarad*cm**-2) * S_N
gl = (5e-5*siemens*cm**-2) * S_N

E_l = -60*mV
# EK = -90*mV
E_Na = 50*mV
g_na = (100*msiemens*cm**-2) * S_N
g_kd = (30*msiemens*cm**-2) * S_N
V_T = -63*mV

# Injected current
I_inj = 20*pamp

# Constants
F = 96485.3321*amp*second/mole

# Astrocyte parameters
S_A = 40e3*umetre**2
Om_A = 5000*umetre**3
g_K = (10*msiemens*cm**-2) * S_A
C_A = (1*ufarad*cm**-2) * S_A

# ECS parameters
Om_E = 10000*umetre**3

# The model
eqs = Equations('''
dv/dt = (gl*(E_l-v)-I_Na-I_K+I_inj)/C_N : volt
I_Na = g_na*m**2*h*(v-E_Na) : amp
I_K = g_kd*n**3*(v-E_K) : amp
dK_i/dt = -I_K/F/Om_N : mmolar
E_K = V_T*log(K_ecs/K_i) : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-v+V_T)/(4*mV))/ms : Hz
beta_m = 0.28*(mV**-1)*5*mV/exprel((v-V_T-40*mV)/(5*mV))/ms : Hz
alpha_h = 0.128*exp((17*mV-v+V_T)/(18*mV))/ms : Hz
beta_h = 4./(1+exp((40*mV-v+V_T)/(5*mV)))/ms : Hz
alpha_n = 0.032*(mV**-1)*5*mV/exprel((15*mV-v+V_T)/(5*mV))/ms : Hz
beta_n = .5*exp((10*mV-v+V_T)/(40*mV))/ms : Hz
K_ecs : mmolar
''')
neuron = NeuronGroup(1, model=eqs, threshold='v>-20*mV', refractory=3*ms,method='euler')
neuron.v = E_l
neuron.K_i = 5*mmolar

eqs = Equations('''
dv/dt = -I_K/C_A : volt
I_K = g_K*(v-E_K) : amp
dK_i/dt = -I_K/Om_A/F : mmolar 
E_K = V_T*log(K_ecs/K_i) : volt
K_ecs : mmolar
''')

astrocyte = NeuronGroup(1, model=eqs, method='euler')
astrocyte.v = 2*E_l
astrocyte.K_i = 0.1*mmolar

eqs = Equations('''
dK_e/dt = I_K_pre/Om_E/F : mmolar
K_ecs_post = K_e : mmolar (summed)
K_ecs_pre = K_e : mmolar (summed)
''')

ecs = Synapses(neuron,astrocyte,model=eqs,method='euler')
ecs.connect('i==j')
ecs.K_e = 1*mmolar

# Allocate some monitors
neu_mon = StateMonitor(neuron, ['v','K_i','E_K'], record=True)
ast_mon = StateMonitor(astrocyte, ['v','K_i','E_K'], record=True)
ecs_mon = StateMonitor(ecs,['K_e'],record=True)
run(1 * second, report='text')
plot(neu_mon.t/ms, neu_mon[0].K_i/mmolar,'k-')
plot(ast_mon.t/ms, ast_mon[0].K_i/mmolar,'r-')
plot(ecs_mon.t/ms, ecs_mon[0].K_e/mmolar,'b-')
show()