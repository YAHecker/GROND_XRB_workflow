#!/usr/bin/env python

import os
import sys
import h5py
import time
import numpy as np
from scipy import constants as const
from scipy.optimize import fsolve
sys.path.append('/home/yahecker/Workflow_generic/')

import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(asctime)s - %(message)s')


def FluxToMag(flux, filtcorr, lambdaeff):
    '''Convert integrated energy flux per unit wavelength F_lambda in a band to 
    AB magnitude in that band.
    
    Parameters:
        flux:           energy flux integrated in band
        filtcorr:       correction factor for integration -- REMNANT OF SIMON'S TESTS, SET
                        THIS TO 1!
        lambdaeff:      effective wavelength of the filter in angstrom (see GROND tables or code below)
    '''
    
    return -2.5*np.log10(flux*filtcorr)-2.402-5.0*np.log10(lambdaeff)


def ParseDefault(default):
    '''Read the parameters in the file <default> for later XRbinary execution.
    Returns a dictionary with XRB keywords as keys and their respective values
    as values. Used for generating a new file for XRB execution.'''
    
    with open(default, 'r') as f:
        lines = f.readlines()

    parameters = {}
    
    for i, line in enumerate(lines):
        
        ### leere Zeilen werden ignoriert
        if '=' in line:
            
            try:
                par = line.split('=')[0]
                val = line.split('=')[1]
                ### if the line has any of these keywords, ignore it
                if par =='COMMENT':
                    continue
                
                elif par =='BANDPASS':
                    continue
                
                elif par=='READDATA':
                    continue
                    
                ### except for this one, for reasons of safety
                elif par=='NORMALIZE':
                    ### ensure that the data is being normalized to the g-band data
                    ### IMPORTANT: This was a choice by me (Yonathan) and should always be
                    ### made for a given system individually. Consult with Jochen in case of
                    ### questions
                    val = '      FITDATA g' 
                    
                parameters[par]=val
                
            except:
                logging.warning(f'Weird behaviour in line {i} of <default>. Ignoring for now.')
                pass
            
    return parameters



def WriteParfile(par_dict, parfile):
    '''Take the dict with XRB parameters and write it into a new file.
    Adds some parameters, such as the filters and the corresponding data
    file paths.'''
    
    with open(parfile, 'w') as f:
        
        for par, val in par_dict.items():
            
            f.write(f'{par}= {val}\n')

        for band in 'g r i z J H K'.split():

            f.write(f'BANDPASS=       FILTER  {band}\n')

            f.write(f'READDATA=       FILTER {band} /home/yahecker/Obs_Data/A0620/flux_{band}.txt\n')
        
        ### ensures that XRB terminates input properly, as otherwise it will crash
        f.write('END\n')


def XRBtoGROND(LCfile, i, T2, TD, rout, rin, T_HS, M_p, q, disctriggers=None):
    '''Take the results computed by XRB and save them in a new
    h5_file. Parameters building the key and being parsed as
    attributes are inlcination <i>, donor temperature <T2>, disc
    temperature at outer edge <TD>, outer disc radius <rout>
    (between 0 and 1) and the inner disc radius <rin>.
    
    Additionally, if there are safety triggers <disctriggers> 
    concerning the disc, they are parsed as an additional attribute.
    '''
    
    ### for converting flux to mags
    ### NOTE: currently not in use in this script, but still neat to have
    # Lambda_mean = np.array([4586, 6220, 7641, 8989, 12398, 16468, 21706])

    ### create h5_file name
    save_key = f'{i}_{T2}_{TD}_{rout}_{rin}'

    ### guard clause to check if the simulation went through properly
    if not os.path.exists(LCfile):

        ### print an output warning for the user
        logging.warning(f"File {save_key} does not exist. Moving on to next simulation!")

        ### delete the file associated with the error to de-clutter the directory
        os.system(f"rm {LCfile.split('.')[0]}")
        
        ### exit, as there is no point in moving on trying to load a non-existing file 
        return
    
    disc_data = np.loadtxt(LCfile, skiprows=5, usecols=(3, 7, 11, 15, 19, 23, 27), unpack=True)
    star_data = np.loadtxt(LCfile, skiprows=5, usecols=(4, 8, 12, 16, 20, 24, 28), unpack=True)
        
    ### Some additional guard clauses concerning the XRB output
    if np.max(star_data) == 0:
    
        logging.warning('Star too dim. Emergency exit!')
            
        raise RuntimeError

    elif np.max(disc_data) == 0 and disctriggers is None:
        
        logging.warning('Disc turned off for some reason. Emergency exit!')
        
        raise RuntimeError
    
    ### initialize empty array to store mags in
    h5_data_star = np.zeros((8, 101), dtype=np.float64)
    h5_data_disc = np.zeros((8, 101), dtype=np.float64)
    
    ### fill empty containers with data
    h5_data_star[0, :] = np.linspace(0, 1, 101)
    h5_data_star[1, :] = star_data[0]
    h5_data_star[2, :] = star_data[1]
    h5_data_star[3, :] = star_data[2]
    h5_data_star[4, :] = star_data[3]
    h5_data_star[5, :] = star_data[4]
    h5_data_star[6, :] = star_data[5]
    h5_data_star[7, :] = star_data[6]
    
    h5_data_disc[0, :] = np.linspace(0, 1, 101)
    h5_data_disc[1, :] = disc_data[0]
    h5_data_disc[2, :] = disc_data[1]
    h5_data_disc[3, :] = disc_data[2]
    h5_data_disc[4, :] = disc_data[3]
    h5_data_disc[5, :] = disc_data[4]
    h5_data_disc[6, :] = disc_data[5]
    h5_data_disc[7, :] = disc_data[6]
    
    ### save lightcurve in h5_file, parse data
    save_location = f'/data/yahecker/A0620/dump/{save_key}.h5'
    
    with h5py.File(save_location, 'a') as h5_file:
        
        if not 'Created on' in list(h5_file.attrs.keys()):
            
            h5_file.attrs['Created on'] = time.ctime()
            h5_file.attrs['Author'] = 'Ascanio Hecker, Y.'
            
        else:
            h5_file.attrs['Modified:'] = time.ctime()
            h5_file.attrs['modified by'] = 'Ascanio Hecker, Y.'
            
        ### attempt to create the set if not already available
        try:
        
            # local_data = h5_file.create_dataset(key, data=h5_data)
            local_star_data = h5_file.create_dataset('Star', data=h5_data_star)
            local_disc_data = h5_file.create_dataset('Disc', data=h5_data_disc)
        
        ### if the dataset exists, overwrite it (with a warning)
        except RuntimeError:
            
            logging.warning(f'Overwriting dataset {save_key}.')

            # h5_file[key] = local_data
            local_star_data = h5_file['Star']
            local_disc_data = h5_file['Disc']
        
        ### do not catch anything else
        except:

            h5_file.close()

            raise
        
        ### Pass some additional attributes, in case you ever get lost with the names
        local_star_data.attrs['Inkl'] = i
        local_star_data.attrs['T_Sec'] = T2
        local_star_data.attrs['M1'] = M_p
        local_star_data.attrs['q'] = q
        local_star_data.attrs['M2'] = np.round(q*M_p, 2)
        
        local_disc_data.attrs['Inkl'] = i
        local_disc_data.attrs['T_Disc'] = TD
        local_disc_data.attrs['f_r'] = rout
        local_disc_data.attrs['R_in'] = np.round(rin, 2)
        local_disc_data.attrs['T_HS'] = T_HS
        
        ### should there be issues with the disc, parse the ocrresponding warning
        if disctriggers is not None:
            # local_data.attrs['Disc_issue'] = disctriggers
            local_disc_data.attrs['Disc_issue'] = disctriggers
            
    ### THIS IS ONLY FOR QUICK REFERENCE OF THE XRB PREDICTION RELATIVE TO THE INTERPOLATED MODEL
    ### LEAVE IT COMMENTED OUT WHEN DOING LARGT GRID CONSTRUCTION OPERATIONS
    # np.savetxt('{}{}{}{}{}'.format(i, T2, TD, rout, rin), np.transpose(disc_data + star_data))
        
    logging.info(f'Saved {save_key} in {save_location}. Proceeding...')

    return
    

def ExecXRB(par_dict, output, i, T_Disc, Dscale, M_p, q, T_s, rin, T_HS, disctrigger):
    '''Starts the XRbinary execution and result saving part of the light curve construction.
    Called by ProduceLC().
    
    Parameters:
        par_dict:           Dictionary with XRB par-val pairs
        output:             temporary file name template for XRB
        i - T_HS:           see later (ProduceLC)
        disctrigger:        Explanatory string for issues arising with the disc at runtime
                            (e.g. too large for XRB boundaries, inner radius > outer radius)
    '''
    
    parfile = f'{output}_tmp'
    LC_xrb_file  = f'{output}_tmp.LC'

    ### directory shenanigans; unnecessary if XRB executable is in the same directory
    # actual_dir = os.getcwd()
    
    workdir = '/home/yahecker/FluxA06/'
    WriteParfile(par_dict, workdir+parfile)
    
    ### continue directory shenanigans
    # os.chdir(workdir)
    
    ### ACTUAL XRbinary EXECUTION HAPPENS HERE
    os.system(f'{workdir}a.out {workdir+parfile}')
    
    ### after XRB is done, call the resaving function
    XRBtoGROND(workdir+LC_xrb_file, i, T_s, T_Disc, Dscale, rin, T_HS, M_p, q, disctrigger)
    
    ### clean up
    os.system(f'rm {workdir+output}_tmp*')
    
    ### concluding directory shenanigans
    # os.chdir(actual_dir)


#### Physical stuff ##########################################

Ms = 1.9891e30 # solar mass in Kg
Rs = 6.96342e8   # solar radius in m
Ts = 5778.0    # sun surface Temperature in Kelvin

G = const.G # gravitational constant in (m^3)/(Kg s^2)


def M2(T):
    '''Estimate donor mass based on surface temperature. Based on stellar scaling
    relations (see e.g. Lang (1992)) and combining a few of these into one equation.
    Temperature in K, mass returned is solar masses.
    
    Note: This is very simplified, potentially include a variable normaizaiton
    factor as second variable and fitting parameter down the line.
    '''
    
    return np.power(T/Ts, 4/2.5)

def fM(K, P):
    '''Returns the mass function value given radial velocity amplitude K in km/s
    and orbital period P in days.
    Return value in units of solar masses.'''
    
    return K**3 * 10**9 * P *24*3600. / (2*np.pi*const.G)/Ms

def mfunc(x, incl, K, P, T):
    '''Dummy callable function for solving the mass function for M1 numerically.
    Parameters:
        x:          M1 [M_Sun]
        incl:       orbital inclination [degrees]
        K:          K_2 of donor star [km/s]
        P:          orbital period [days]
        T:          donor surface temperature [K]'''
    
    f = fM(K, P)
    sinthird = np.sin(incl/180. *np.pi)**3
    
    return f*(1 + M2(T) / x)**2 / sinthird - x

def M1(incl, K, P, T):
    '''Wrapper for solving the mass function for M1 numerically.'''
    
    sol = fsolve(mfunc, 7*np.ones_like(incl), args = (incl, K, P, T))
    
    return sol[0]

def mfunc_new(x, incl, K, P, M_2):
    '''Alternative dummy callable, now with donor mass M_2 instead of temperature.'''

    f = fM(K, P)
    sinthird = np.sin(incl/180. *np.pi)**3
    
    return f*(1 + M_2 / x)**2 / sinthird - x

def mfunc2(x, incl, K, P, q):
    '''Yet another dummy callable, this time with a known mass ratio q.'''
    
    f = fM(K, P)
    sinthird = np.sin(incl/180. *np.pi)**3
    
    return f*(1 + q)**2 / sinthird - x

### Primary Mass solving mass function numerically for inclination
def M1_new(incl, K, P, M_2):
    '''Wrapper for solving mass function for M1 numerically given M_2.'''

    sol = fsolve(mfunc_new, 7*np.ones_like(incl), args = (incl, K, P, M_2))
    
    return sol[0]

def M1_BW(incl, K, P, q):
    '''Wrapper for solving mass function for M1 numerically given q.'''
    
    sol = fsolve(mfunc2, 7*np.ones_like(incl), args = (incl, K, P, q))    
    
    return sol[0]


### Distance of objects
def a(M_star, M_bh, P):
    '''Calculate the orbital distance between the donor and the BH given the orbital period.
    Return units are solar radii.'''

    return (  ( (G*(M_star+M_bh)*Ms*(P*24*60*60)**2) /(4*np.pi**2) )**(1.0/3.0)  ) /Rs

### ### Radius Accretion disc Shakura & Sunyaev 1973 (cannot confirm this source, but alas...)
### def rd(M_star, M_bh, P):
###     q = M_star/M_bh
###     return (0.6* a(M_star, M_bh, P))/(1+q)

### # circulation radius (=minimum outer radius) for 0.1 < q < 10 (Could not find the source anymore, as Frank et al. 2002 is
### # no longer available in the library)
### # may be more accurate than the equation from H&H 1995, but I don't like citing stuff that I don't know the origin of
### def r_circ(M_star, M_bh):
###     q = M_star/M_bh
###     return((1+q)*(0.5-0.227*np.log10(q))**4)

### from H&H 1995; 
def r_circ(M_star, M_bh):
    '''Return the circularisation radius in multiples of the orbital separation.'''
    
    q = M_star / M_bh
    
    return 0.0859*np.power(q, -0.426)

### tidal disruption radius (Frank et al. 2002, Eggleton 1983)
def rd_t(M_star, M_bh):
    '''Return the tidal disruption radius in multiples of the orbital separation.'''
    
    q = M_star/M_bh
    
    return 0.9*0.49*q**(-2/3.)/(0.6*q**(-2/3.)+np.log10(1+q**(-1/3.)))

def R_ISCO(M_bh):
    '''Return the innermost stable circular orbit (ISCO) around the Schwarzschild BH.
    Give BH mass in solar masses. Returns meters.'''
    
    return(3*2*M_bh*Ms*G/const.c**2)

def ProduceLC(period, T_s, i, DiscOnOff, DiscrimOnOff, T_Disc, phaseoffset, Dscale, DiscSpotOnOff, Rinner, T_HS, File):
    '''Wrapper to construct a set of light curves in XRbinary given several input parameters.'''
    
    workdir = '/home/yahecker/FluxA06/'
    
    ### get the parameter keywords necessary for XRB
    parameters =   ParseDefault(workdir+'default')

    T_pow = -0.75

    M_2 = M2(T_s)
    
    ### Calculate BH mass given P, K, i and M2
    M_p = M1_new(i, 437.1, period, M_2)
    
    # we're going to need this anyway
    q = M_2/M_p

    ### THIS IS ONLY FOR BW CIR, AS IT IS NOT ON THE MAIN SEQUENCE ###
    # q = 0.12
    # 
    # M_p = M1_BW(i, 279.0, period, q)
    # 
    # M_2 = M_p * q
    ##################################################################

    distance = a(M_2, M_p, period) #distance objects [solar radii]
    
    max_radius = rd_t(M_2, M_p) #disc disruption radius [orbital distances]
    
    min_radius = r_circ(M_2, M_p)  # disc circularisation radius [orbital distances]
    
    ### scale outer disc radius by 
    A_Disc = min_radius + Dscale * (max_radius - min_radius) 

    # A_Disc = Dscale
    
    ### Trigger to check intervention at outer disc edge
    out_trigger = None

    ### XRbinary implemented the 0.6*a upper limit for the disc. Why? I don't know (yet) | Yonathan
    if A_Disc > 0.6:
        out_trigger = 'Outer radius truncated by XRB.'
        A_Disc = 0.6
    
    ### check whether under the current assumptions the outer radius is consistent with ongoing accretion and physics
    if A_Disc > max_radius:
        out_trigger = 'Outer radius inconsistent with r_td. '
    elif A_Disc < min_radius:
        out_trigger = 'Outer radius inconsistent with r_circ. '
    
    A_minDisc = Rinner * R_ISCO(M_p) / 3. / (distance * Rs)

    # A_minDisc = Rinner
    
    ### Trigger to checkintervention at inner disc edge
    in_trigger = None
    
    if not A_Disc > A_minDisc:
        ### worst case: the inner radius is larger than the outer radius -> no disc
        DiscOnOff = 'OFF'
        in_trigger = 'Inner radius larger than outer radius. Disc disabled.'

    if A_minDisc > min_radius:
        ### another really bad case: the inner radius is larger than the smallest possible outer radius
        DiscOnOff = 'OFF'
        in_trigger = 'Inner radius larger than circularisation radius. Disc disabled.'
    
    Hedge = 0.1 * (A_Disc - A_minDisc) #maximum
    
    ### Set the edge temperature to the temperature at the outer disc
    T_DiscEdge = T_Disc   
    
        
    ### save proper parameter values in XRB-compatible format
    parameters['INCLINATION']= '%s'%i
    parameters['K2'] = '437.1'
    parameters['MASSRATIO']= '%s'%q
    parameters['STAR2TEMP']= '%s'%T_s
    parameters['IRRADIATION']= 'ON'
    
    parameters['PERIOD'] = '%s'%period
    parameters['PHASEOFFSET']= '%s'%phaseoffset
    
    parameters['INNERDISK']= 'OFF'
    parameters['DISK']= DiscOnOff
    parameters['DISKSPOTS']= DiscSpotOnOff
    parameters['DISKTORUS']= DiscrimOnOff
    parameters['DISKEDGET']= '%s  %s  %s  %s'%(T_DiscEdge, T_DiscEdge, 90, 10)   ### NO HOTSPOT ON EDGE AT THE MOMENT
    ### this could be changed to a steady-state visous disc by changint the value to 'VISCOUS <T_Disc>'
    parameters['MAINDISKT']= '%s  %s'%(T_Disc, T_pow)    
    parameters['MAINDISKA']= '%s  %s'%(A_minDisc, A_Disc)
    parameters['MAINDISKH']= '%s 1.125'%(Hedge)  #r**9/8 from Frank et al. 2002
    
    trigger = None
    # ExecXRB(parameters, File, i, T_Disc, Dscale, M_p, M_s, q, T_s, A_minDisc, T_HS, out_trigger, in_trigger)
    if out_trigger is not None:
        if in_trigger is not None:
            trigger = out_trigger + 'Additionally: ' + in_trigger
        else:
            trigger = out_trigger
    elif in_trigger is not None:
        trigger = in_trigger
    
    ### user information; you can also delete this, if you so feel
    logging.info(f'Simulating LC: i={i}, T_s={T_s}, T_Disc={T_Disc}, Dscale={Dscale}, Rinner={Rinner}, T_HS={T_HS}; overlapping radii: {trigger}')
    
    ### calling actual XRB execution and saving routines afterwards
    ExecXRB(parameters, File, i, T_Disc, Dscale, M_p, q, T_s, Rinner, T_HS, trigger)

    ### reset the disk state
    if in_trigger:
        DiscOnOff = 'ON'
    
    ### reset the triggers and Disk state if necessary
    if in_trigger == True:
        DiskOnOff = original_DiskOnOff
        in_trigger = False
    
    out_trigger = False
    
########################################## Main Programm #################################################

def building_block(inlist):
    '''Takes a list of values for i, f_r, R_in, T_2 and constructs lightcurves for all 
    combinatinos of these parameters with other parameters' provided ranges.
    NOTE: Poor design, but it works.'''
    
    Inkl, Dscale, R_inner, T_s = inlist
    
    for T_Disc in Trange:
        
        for T_hs in THSrange:
        
            filename = f'{Inkl}_{T_s}_{T_Disc}_{int(100*Dscale)}_{R_inner}_{T_hs}' ### f_r, k*R_ISCO parameterisation
            
            ### if you have already started a version of the grid and it broke in the end, you can use this line to
            ### skip points that are already constructed, not overwriting them with redundant information
            
            # if not os.path.exists(f'/ptmp/yahecker/Grids/A0620/{filename}.h5'):
            ProduceLC(period, T_s, Inkl, DiscOnOff, DiscrimOnOff, T_Disc, phaseoffset, Dscale, DiscSpotOnOff, R_inner, T_hs, filename)

## Specify
period = 0.32301405
phaseoffset = 0
DiskrimOnOff = 'OFF'
DiskOnOff = 'ON'
HsOnOff   = 'ON'
DiskSpotOnOff = 'OFF'


## Modify
Inkl = -10
T_Sec = 1e8
T_Disc = -10
f_r = 2
R_inner = 0
T_HS = 1e8

ProduceLC(period, T_Sec, Inkl, DiskOnOff, DiskrimOnOff, T_Disc, phaseoffset, f_r, DiskSpotOnOff, R_inner, T_HS, "{}{}{}{}{}".format(Inkl, T_Sec, T_Disc, f_r, R_inner))

