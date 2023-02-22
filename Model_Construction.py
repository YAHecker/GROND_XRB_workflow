#!/usr/bin python

from astromodels import Band
from astromodels import TemplateModel
from astromodels import TemplateModelFactory
import numpy as np
import string
from scipy import constants as const
from time import sleep
import sys
sys.path.append('/home/yahecker/FluxA06/')
from my_functions import A06_f_model_no_HS
from progress.bar import Bar

def modfactory(band, col, AV, shift):
    phases = np.arange(0, 1.01, 0.01)
    print(band, col, AV)
    tmf = TemplateModelFactory('A06_f_no_HS_Knorm_%s_%s'%(int(AV*10), band), 
                               'Template model with free disk, f as outer radius parameter and AV=%smag - %s, normalized to the K-band data. Phase shift of %.2f applied!'%(AV, band, shift), 
                               phases, 
                               ['Inkl', 'T_Sec', 'T_Disc', 'f_r', 'R_inner'])

    Inkl_grid = np.arange(50, 75, 4)
    T_s_grid = np.arange(3800, 5001, 200)
    T_d_grid = np.arange(2500, 3901, 200)
    f_range = np.arange(0, 1.01, 0.1)
    Rinner_range = np.arange(1000, 10001, 1000)
   
    tmf.define_parameter_grid('Inkl', Inkl_grid)    
    tmf.define_parameter_grid('T_Sec', T_s_grid)
    tmf.define_parameter_grid('T_Disc', T_d_grid)
    tmf.define_parameter_grid('f_r', f_range)
    tmf.define_parameter_grid('R_inner', Rinner_range)
    zeiger = Bar('Processing', max = np.shape(Inkl_grid)[0]*np.shape(T_s_grid)[0]*np.shape(T_d_grid)[0]*np.shape(Rinner_range)[0]*len(f_range))
    
    for a in Inkl_grid:
        for b in T_s_grid:
            for c in T_d_grid:
                for d in f_range:
                    for e in Rinner_range:
                        tmf.add_interpolation_data(A06_f_model_no_HS(phases, a, b, c, d, e, AV, col, shift), Inkl = a, T_Sec = b, T_Disc = c, f_r = d, R_inner = e)
                        zeiger.next()
    tmf.save_data(overwrite=True)
    zeiger.finish()
    print('file saved')

filter_list = ['gband', 'rband', 'iband', 'zband', 'Jband', 'Hband', 'Kband']

shift = -0.06
AV_range = [0.5]
for Av in AV_range:
    for col, band in enumerate(filter_list):
        modfactory(filter_list[col], col, Av, shift)
