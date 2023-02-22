#!/usr/bin/env python
from threeML import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import time
import matplotlib
matplotlib.use('agg')
plt.rcParams.update({'font.size':14, 'figure.figsize':(12, 7)})

def ModeltoPointSource(band):
    '''Load the interpolated XRbinary light curve grid that
    has the components for the star and the disc seperately.
    Returns a PointSource instance for use in a Model instance
    later.'''
    
    star = TemplateModel('A06_f_noHS_star_%s'%band, log_interp = False)
    star.K.fix = False
    star.scale.fix = True
    
    disc = TemplateModel('A06_f_noHS_disc_%s'%band, log_interp = False)
    disc.K.fix = False
    disc.scale.fix = True
    
    # print(f'Loaded {band} model.')
    
    ### Positions a source with the name '<band>_source' on the sky with components 
    ### "Star" and "Disc", which get summed up. Hence, the units are energy flux per
    ### unit area, wavelength and time
    return PointSource('%s_source'%band, 0, 0, components=[SpectralComponent('Star', star), SpectralComponent('Disc', disc)])

class FilterbandExtinctionFunction(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        A function for calculating the extinction over a band with a pivot band
    latex : $ F_\lambda = K \cdot 10^{-0.4 \cdot k_{band} \cdot A_V} $
    parameters :
        K :
            desc : Normalization (differential flux without extinction)
            initial value : 1.0
            is_normalization : True
            fix : True
            min : 1e-30
            max : 1e3
            delta : 0.1
        k_band :
            desc : band dependent ratio of extinction in this band relative to Johnson V band ( $ A_\lambda/A_V $ )
            initial value : 1.0
            fix : yes
            min : -2
            max : 5
    """
    
    def _set_units(self, x_unit, y_unit):
        # The extinction constants are all dimensionless
        self.k_band.unit = u.dimensionless_unscaled

        # The normalization has the same units as the flux

        self.K.unit = y_unit


    def evaluate(self, x, K, k_band):

        return K * np.power(10, -0.4 * k_band * x)



if __name__ == '__main__':
    
    ### load the data for all bands individually
    pa = '/home/yahecker/Obs_Data/A0620/'
    data_path = pa + 'flux_%s_fit.txt'%'g_red'
    gdata = XYLike.from_text_file("gdata", data_path)
    ### and assign them a source for the fit later
    gdata.assign_to_source(source_name = 'gband_source')

    data_path = pa +'flux_%s_fit.txt'%'r_red'
    rdata = XYLike.from_text_file("rdata", data_path)
    rdata.assign_to_source(source_name = 'rband_source')

    data_path = pa +'flux_%s_fit.txt'%'i_red'
    idata = XYLike.from_text_file("idata", data_path)
    idata.assign_to_source(source_name = 'iband_source')

    data_path = pa +'flux_%s_fit.txt'%'z_red'
    zdata = XYLike.from_text_file("zdata", data_path)
    zdata.assign_to_source(source_name = 'zband_source')

    data_path = pa +'flux_%s_fit.txt'%'J_red'
    Jdata = XYLike.from_text_file("Jdata", data_path)
    Jdata.assign_to_source(source_name = 'Jband_source')

    data_path = pa +'flux_%s_fit.txt'%'H_red'
    Hdata = XYLike.from_text_file("Hdata", data_path)
    Hdata.assign_to_source(source_name = 'Hband_source')

    data_path = pa +'flux_%s_fit.txt'%'K_red'
    Kdata = XYLike.from_text_file("Kdata", data_path)
    Kdata.assign_to_source(source_name = 'Kband_source')

    ### create a DataList object with all the components' data
    data = DataList(gdata, rdata, idata, zdata, Jdata, Hdata, Kdata)


    ### GROND band multiplication factors for the extinction (A_band = x * A_V)
    extinction_factors = {'g':1.282,
                          'r':0.897,
                          'i':0.641,
                          'z':0.513,
                          'J':0.385,
                          'H':0.256,
                          'K':0.128
                         }
    ### NOTE: These can somewhat depend on the input spectrum, so they may be
    ### off for the sources in question... | Yonathan    

    ### Create the actual Model instance containing all the sources
    filter_list = ['gband', 'rband', 'iband', 'zband', 'Jband', 'Hband', 'Kband']
    point_source_list = ['ps_g', 'ps_r', 'ps_i', 'ps_z', 'ps_J', 'ps_H', 'ps_K']
    PointSdict = {}
    for h in range(len(filter_list)):
        PointSdict[point_source_list[h]] = ModeltoPointSource(filter_list[h])

    my_model = Model(PointSdict['ps_g'], PointSdict['ps_r'],
                     PointSdict['ps_i'], PointSdict['ps_z'],
                     PointSdict['ps_J'], PointSdict['ps_H'], PointSdict['ps_K'])

    ### Add the extinction to the model as a parameter
    parameter_Extinction = Parameter(name='A_V',
                                     value=0.5,
                                     min_value=0,
                                     max_value=2,
                                     free=True,
                                     unit='',
                                     delta=0.05)

    my_model.add_external_parameter(parameter_Extinction)

    
    ### link the extinction parameter to each band normalization using the appropriate
    ### relation each time; in this case, XRbinary normalized to the g band, so its
    ### factor needs to be subtracted from the band factors in the exponent (see thesis)
    my_model.link(my_model.gband_source.spectrum.Star.A06_f_noHS_star_gband.K, my_model.A_V, FilterbandExtinctionFunction(k_band=extinction_factors['g'] - extinction_factors['g']))
    
    my_model.link(my_model.rband_source.spectrum.Star.A06_f_noHS_star_rband.K, my_model.A_V, FilterbandExtinctionFunction(k_band=extinction_factors['r'] - extinction_factors['g']))
    
    my_model.link(my_model.iband_source.spectrum.Star.A06_f_noHS_star_iband.K, my_model.A_V, FilterbandExtinctionFunction(k_band=extinction_factors['i'] - extinction_factors['g']))
    
    my_model.link(my_model.zband_source.spectrum.Star.A06_f_noHS_star_zband.K, my_model.A_V, FilterbandExtinctionFunction(k_band=extinction_factors['z'] - extinction_factors['g']))
    
    my_model.link(my_model.Jband_source.spectrum.Star.A06_f_noHS_star_Jband.K, my_model.A_V, FilterbandExtinctionFunction(k_band=extinction_factors['J'] - extinction_factors['g']))
    
    my_model.link(my_model.Hband_source.spectrum.Star.A06_f_noHS_star_Hband.K, my_model.A_V, FilterbandExtinctionFunction(k_band=extinction_factors['H'] - extinction_factors['g']))
    
    my_model.link(my_model.Kband_source.spectrum.Star.A06_f_noHS_star_Kband.K, my_model.A_V, FilterbandExtinctionFunction(k_band=extinction_factors['K'] - extinction_factors['g']))
    
    my_model.link(my_model.gband_source.spectrum.Disc.A06_f_noHS_disc_gband.K, my_model.A_V, FilterbandExtinctionFunction(k_band=extinction_factors['g'] - extinction_factors['g']))
    
    my_model.link(my_model.rband_source.spectrum.Disc.A06_f_noHS_disc_rband.K, my_model.A_V, FilterbandExtinctionFunction(k_band=extinction_factors['r'] - extinction_factors['g']))
    
    my_model.link(my_model.iband_source.spectrum.Disc.A06_f_noHS_disc_iband.K, my_model.A_V, FilterbandExtinctionFunction(k_band=extinction_factors['i'] - extinction_factors['g']))
    
    my_model.link(my_model.zband_source.spectrum.Disc.A06_f_noHS_disc_zband.K, my_model.A_V, FilterbandExtinctionFunction(k_band=extinction_factors['z'] - extinction_factors['g']))
    
    my_model.link(my_model.Jband_source.spectrum.Disc.A06_f_noHS_disc_Jband.K, my_model.A_V, FilterbandExtinctionFunction(k_band=extinction_factors['J'] - extinction_factors['g']))
    
    my_model.link(my_model.Hband_source.spectrum.Disc.A06_f_noHS_disc_Hband.K, my_model.A_V, FilterbandExtinctionFunction(k_band=extinction_factors['H'] - extinction_factors['g']))
    
    my_model.link(my_model.Kband_source.spectrum.Disc.A06_f_noHS_disc_Kband.K, my_model.A_V, FilterbandExtinctionFunction(k_band=extinction_factors['K'] - extinction_factors['g']))


    ### Prepare to link all other parameters to one instance, ensuring consistency 
    ### across the model
    ### NOTE: this list does NOT contain the verison of Inkl it's supposed to be linked to
    allInklParams =[my_model.rband_source.spectrum.Star.A06_f_noHS_star_rband.Inkl,
                    my_model.iband_source.spectrum.Star.A06_f_noHS_star_iband.Inkl,
                    my_model.zband_source.spectrum.Star.A06_f_noHS_star_zband.Inkl,
                    my_model.Jband_source.spectrum.Star.A06_f_noHS_star_Jband.Inkl,
                    my_model.Hband_source.spectrum.Star.A06_f_noHS_star_Hband.Inkl,
                    my_model.Kband_source.spectrum.Star.A06_f_noHS_star_Kband.Inkl,
                    my_model.gband_source.spectrum.Disc.A06_f_noHS_disc_gband.Inkl,
                    my_model.rband_source.spectrum.Disc.A06_f_noHS_disc_rband.Inkl,
                    my_model.iband_source.spectrum.Disc.A06_f_noHS_disc_iband.Inkl,
                    my_model.zband_source.spectrum.Disc.A06_f_noHS_disc_zband.Inkl,
                    my_model.Jband_source.spectrum.Disc.A06_f_noHS_disc_Jband.Inkl,
                    my_model.Hband_source.spectrum.Disc.A06_f_noHS_disc_Hband.Inkl,
                    my_model.Kband_source.spectrum.Disc.A06_f_noHS_disc_Kband.Inkl]

    allT_SecParams = [my_model.rband_source.spectrum.Star.A06_f_noHS_star_rband.T_Sec,
                      my_model.iband_source.spectrum.Star.A06_f_noHS_star_iband.T_Sec,
                      my_model.zband_source.spectrum.Star.A06_f_noHS_star_zband.T_Sec,
                      my_model.Jband_source.spectrum.Star.A06_f_noHS_star_Jband.T_Sec,
                      my_model.Hband_source.spectrum.Star.A06_f_noHS_star_Hband.T_Sec,
                      my_model.Kband_source.spectrum.Star.A06_f_noHS_star_Kband.T_Sec,
                      my_model.gband_source.spectrum.Disc.A06_f_noHS_disc_gband.T_Sec,
                      my_model.rband_source.spectrum.Disc.A06_f_noHS_disc_rband.T_Sec,
                      my_model.iband_source.spectrum.Disc.A06_f_noHS_disc_iband.T_Sec,
                      my_model.zband_source.spectrum.Disc.A06_f_noHS_disc_zband.T_Sec,
                      my_model.Jband_source.spectrum.Disc.A06_f_noHS_disc_Jband.T_Sec,
                      my_model.Hband_source.spectrum.Disc.A06_f_noHS_disc_Hband.T_Sec,
                      my_model.Kband_source.spectrum.Disc.A06_f_noHS_disc_Kband.T_Sec]

    allT_DiscParams = [my_model.rband_source.spectrum.Star.A06_f_noHS_star_rband.T_Disc,
                       my_model.iband_source.spectrum.Star.A06_f_noHS_star_iband.T_Disc,
                       my_model.zband_source.spectrum.Star.A06_f_noHS_star_zband.T_Disc,
                       my_model.Jband_source.spectrum.Star.A06_f_noHS_star_Jband.T_Disc,
                       my_model.Hband_source.spectrum.Star.A06_f_noHS_star_Hband.T_Disc,
                       my_model.Kband_source.spectrum.Star.A06_f_noHS_star_Kband.T_Disc,
                       my_model.gband_source.spectrum.Star.A06_f_noHS_star_gband.T_Disc,
                       my_model.rband_source.spectrum.Disc.A06_f_noHS_disc_rband.T_Disc,
                       my_model.iband_source.spectrum.Disc.A06_f_noHS_disc_iband.T_Disc,
                       my_model.zband_source.spectrum.Disc.A06_f_noHS_disc_zband.T_Disc,
                       my_model.Jband_source.spectrum.Disc.A06_f_noHS_disc_Jband.T_Disc,
                       my_model.Hband_source.spectrum.Disc.A06_f_noHS_disc_Hband.T_Disc,
                       my_model.Kband_source.spectrum.Disc.A06_f_noHS_disc_Kband.T_Disc]

    allf_rParams = [my_model.rband_source.spectrum.Star.A06_f_noHS_star_rband.f_r,
                    my_model.iband_source.spectrum.Star.A06_f_noHS_star_iband.f_r,
                    my_model.zband_source.spectrum.Star.A06_f_noHS_star_zband.f_r,
                    my_model.Jband_source.spectrum.Star.A06_f_noHS_star_Jband.f_r,
                    my_model.Hband_source.spectrum.Star.A06_f_noHS_star_Hband.f_r,
                    my_model.Kband_source.spectrum.Star.A06_f_noHS_star_Kband.f_r,
                    my_model.gband_source.spectrum.Star.A06_f_noHS_star_gband.f_r,
                    my_model.rband_source.spectrum.Disc.A06_f_noHS_disc_rband.f_r,
                    my_model.iband_source.spectrum.Disc.A06_f_noHS_disc_iband.f_r,
                    my_model.zband_source.spectrum.Disc.A06_f_noHS_disc_zband.f_r,
                    my_model.Jband_source.spectrum.Disc.A06_f_noHS_disc_Jband.f_r,
                    my_model.Hband_source.spectrum.Disc.A06_f_noHS_disc_Hband.f_r,
                    my_model.Kband_source.spectrum.Disc.A06_f_noHS_disc_Kband.f_r]

    allRinnerParams = [my_model.rband_source.spectrum.Star.A06_f_noHS_star_rband.R_in,
                       my_model.iband_source.spectrum.Star.A06_f_noHS_star_iband.R_in,
                       my_model.zband_source.spectrum.Star.A06_f_noHS_star_zband.R_in,
                       my_model.Jband_source.spectrum.Star.A06_f_noHS_star_Jband.R_in,
                       my_model.Hband_source.spectrum.Star.A06_f_noHS_star_Hband.R_in,
                       my_model.Kband_source.spectrum.Star.A06_f_noHS_star_Kband.R_in,
                       my_model.gband_source.spectrum.Star.A06_f_noHS_star_gband.R_in,
                       my_model.rband_source.spectrum.Disc.A06_f_noHS_disc_rband.R_in,
                       my_model.iband_source.spectrum.Disc.A06_f_noHS_disc_iband.R_in,
                       my_model.zband_source.spectrum.Disc.A06_f_noHS_disc_zband.R_in,
                       my_model.Jband_source.spectrum.Disc.A06_f_noHS_disc_Jband.R_in,
                       my_model.Hband_source.spectrum.Disc.A06_f_noHS_disc_Hband.R_in,
                       my_model.Kband_source.spectrum.Disc.A06_f_noHS_disc_Kband.R_in]

    ### actual linking process; here, the remaining 5 versions can be found
    my_model.link(allInklParams, my_model.gband_source.spectrum.Star.A06_f_noHS_star_gband.Inkl)
    my_model.link(allT_SecParams, my_model.gband_source.spectrum.Star.A06_f_noHS_star_gband.T_Sec)
    my_model.link(allT_DiscParams, my_model.gband_source.spectrum.Disc.A06_f_noHS_disc_gband.T_Disc)
    my_model.link(allRinnerParams, my_model.gband_source.spectrum.Disc.A06_f_noHS_disc_gband.R_in)
    my_model.link(allf_rParams, my_model.gband_source.spectrum.Disc.A06_f_noHS_disc_gband.f_r)


    ### Define priors for each parameter
    my_model.gband_source.spectrum.Star.A06_f_noHS_star_gband.Inkl.prior = Truncated_gaussian(mu = 58,
                                                                                         sigma = 15,
                                                                                         lower_bound = 46,
                                                                                         upper_bound=74)

    my_model.gband_source.spectrum.Star.A06_f_noHS_star_gband.T_Sec.prior = Truncated_gaussian(mu = 4500,
                                                                                          sigma=800,
                                                                                          lower_bound = 3750,
                                                                                          upper_bound = 5250)

    my_model.gband_source.spectrum.Disc.A06_f_noHS_disc_gband.T_Disc.prior = Truncated_gaussian(mu = 2750,
                                                                                          sigma=1000,
                                                                                          lower_bound = 1000,
                                                                                          upper_bound = 4500)

    my_model.gband_source.spectrum.Disc.A06_f_noHS_disc_gband.f_r.prior = Truncated_gaussian(mu = 0.3,
                                                                                          sigma = 0.6,
                                                                                          lower_bound = 0.0,
                                                                                          upper_bound = 1.0)

    my_model.gband_source.spectrum.Disc.A06_f_noHS_disc_gband.R_in.prior = Truncated_gaussian(mu = 6000,
                                                                                            sigma=4000,
                                                                                            lower_bound = 1000,
                                                                                            upper_bound = 10000)

    my_model.A_V.prior = Truncated_gaussian(mu = 1.0, sigma=0.5, lower_bound = 0, upper_bound=2.0)
    
    ### In case you need to fix any quantity for your fit, this is how you would do it
    # my_model.gband_source.spectrum.Star.A06_f_noHS_star_gband.T_Sec.value = 4600
    # my_model.gband_source.spectrum.Star.A06_f_noHS_star_gband.T_Sec.fix = True
    

    with use_astromodels_memoization(False):

        ba = BayesianAnalysis(my_model, data)

        ba.set_sampler('emcee')

        ba.sampler.setup(n_iterations=500, n_burn_in=3000, n_walkers=50, seed=51338467)

        ba.sample()

        ba.results.write_to('Fitting_Results/test.fits', overwrite=True)

    
    ba.results.corner_plot()
    plt.savefig(f'plots/test.pdf', format='pdf')

