import numpy as np 
import scipy as sp 
try:
    from .LSDparse import parse_LSDERA40
except ImportError:
    from LSDparse import parse_LSDERA40

def convert_xyz_to_pressure(
    site_lat : float, 
    site_lon : float, 
    site_elev : float,
    era40 : dict = parse_LSDERA40()
):
    
    # Looks up mean sea level pressure and mean 1000 mb temp from ERA-40 reanalysis
    # and calculates site atmospheric pressures using these as inputs to the
    # standard atmosphere equation. 
    #
    # Syntax: pressure = ERA40(site_lat,site_lon,site_elv)
    # 
    # Requires:
    #       site_lat: latitude (DD). Southern hemisphere is negative.
    #       site_lon: longitude (DD). Western hemisphere is negative.
    #           Tries to deal with 0-360 longitudes gracefully.
    #       site_elv: elevation (m).
    #
    # Returns site pressure in hPa.
    #
    # Vectorized. Send vectors of equal length.
    #
    # Note: this must load the data file ERA40.mat whenever called. 
    # Repeated calls to this function will be slow for this reason. 
    #
    # Also: This function is OK but not great for Antarctica.
    # Use antatm.m instead. 
    #
    # Remember: it is always better to estimate the average pressure at your 
    # site using a pressure-altitude relation obtained from nearby station
    # data.
    #

    # Written by Greg Balco -- UW Cosmogenic Nuclide Lab
    # balcs@u.washington.edu
    # October, 2007
    # Part of the CRONUS-Earth online calculators: 
    #      http://hess.ess.washington.edu/math
    #
    # Copyright 2001-2007, University of Washington
    # All rights reserved
    # Developed in part with funding from the National Science Foundation.
    # 
    # Modified by Nat Lifton -- Purdue University to use ERA-40 instead of NCEP
    # March 2011

    # This program is free software you can redistribute it and/or modify
    # it under the terms of the GNU General Public License, version 3,
    # as published by the Free Software Foundation (www.fsf.org).
    
    # correct negative longitudes

    if site_lon < 0: site_lon += 360

    # Interpolate sea level pressure and 1000-mb temperature
    # from global reanalysis data grids. 

    # site_T in K, site_P in hPa

    slp_interpolator = sp.interpolate.RegularGridInterpolator(
        (era40["ERA40lat"], era40["ERA40lon"]), 
        era40["meanP"],
        method="linear",
        bounds_error=True
    )
    site_slp = slp_interpolator((site_lat, site_lon))
    T_interpolator = sp.interpolate.RegularGridInterpolator(
        (era40["ERA40lat"], era40["ERA40lon"]), 
        era40["meanT"],
        method="linear",
        bounds_error=True
    )
    site_T = T_interpolator((site_lat, site_lon))

    # site_slp = 1013.25
    # site_T = 288.15
    # 
    # site_T_degK = site_T + 273.15

    # More parameters

    gmr = -0.03417 # Assorted constants
    # dtdz = 0.0065 # Lapse rate from standard atmosphere

    # Lifton Lapse Rate Fit to COSPAR CIRA-86 <10 km altitude

    lr = np.array([-6.1517E-03, -3.1831E-06, -1.5014E-07, 1.8097E-09, 1.1791E-10, -6.5359E-14, -9.5209E-15])

    dtdz = lr[0] + lr[1]*site_lat + lr[2]*site_lat**2 + \
        lr[3]*site_lat**3 + lr[4]*site_lat**4 + lr[5]* site_lat**5 + \
        lr[6]*site_lat**6
    dtdz = -dtdz

    # Variable lapse rate experiments -- attempts to make lapse rate a 
    # physically reasonable function of temperature. No guarantees on the 
    # correctness of this part. 
    #
    # Not used in final version. Probably more physically correct but 
    # has a very limited effect on the overall results. Code retained here as
    # comments for those who are interested. If you are an atmospheric
    # scientist, feel free to suggest a better way to do this. 
    # 
    # Temp assumed for lapse rate is 15 degrees less than real temp.
    # Chosen because in stdatm, std T is 15 and LR is MALR for 0 deg C.
    # Not sure why this is the case. Ask an atmospheric scientist. 
    #
    #Tlr = site_T_degK - 15
    #
    # Calculate the saturation vapor pressure
    #
    #esat = 10*0.6112*exp(17.67*(Tlr-273.15)./((Tlr-273.15)+243.5))
    #
    # Calculate the mixing ratio and thence the lapse rate
    #
    #rv = 0.622*esat./(1013.25-esat)
    #Lv = 2.501e6 R = 287 E = 0.62 g = 9.8066 cpd = 1005.7
    #dtdz = g*(1+(Lv*rv./R./Tlr))./(cpd + (Lv**2*rv*E./R./(Tlr**2)))

    # Calculate site pressure using the site-specific SLP and T1000 with the
    # standard atmosphere equation.

    out = site_slp * np.exp( (gmr/dtdz) * ( np.log(site_T) - np.log(site_T - (site_elev*dtdz)) ) )
    
    return out

if __name__ == "__main__": 
    out = convert_xyz_to_pressure(
        1,
        1,
        1
    )
    print(out)
    
    
    