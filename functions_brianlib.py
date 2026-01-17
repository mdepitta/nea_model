import scipy.constants as spc
from brian2 import *

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

def HeavisideFunction(x,eps):
    """
    Continuous implementation of the Heaviside function

    Input:
    - x   : float
    - eps : slope of the step

    Return:
    - heavisde(x)
    """
    if eps<=0.0:
       return 1.0 if x>=0.0 else 0.0
    else:
        z = x/eps
        if (z>=0.0):
            e = np.exp(-z)
            return 1.0/(1.0+e)
        else:
            e = np.exp(z)
            return e/(1.0+e)
    # return np.abs(x)/(np.abs(x)+eps)
HeavisideFunction = Function(HeavisideFunction,arg_units=[1,1], return_unit=1,auto_vectorise=False)
HeavisideFunction_cpp = '''
    #include <gsl/gsl_const_mksa.h>
    double HeavisideFunction(double x, const double eps)
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
HeavisideFunction.implementations.add_implementation('cpp',HeavisideFunction_cpp,
                                                     dependencies={'exp': DEFAULT_FUNCTIONS['exp']})

# if __name__=="__main__":
#     import matplotlib.pyplot as plt
#     x_ = np.concatenate((np.linspace(-10,0),np.linspace(0.1,10,99)))
#     eps = 1e-1
#     ivb = np.zeros_like(x_)
#     for i,v in enumerate(x_):
#         ivb[i] = HeavisideFunction(v,eps)
#     plt.plot(x_,ivb,'k-')
#     plt.show()