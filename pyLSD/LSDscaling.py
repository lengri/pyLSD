import numpy as np
import scipy as sp
import os 
try:
    from .LSDparse import parse_LSDconsts, parse_LSDERA40
    from .LSDspectra import (
        calculate_muon_flux,
        calculate_proton_flux,
        calculate_neutron_flux,
        calculate_low_E_neutron_flux
    )
    from .LSDatm import convert_xyz_to_pressure
except ImportError:
    from LSDparse import parse_LSDconsts, parse_LSDERA40
    from LSDspectra import (
        calculate_muon_flux,
        calculate_proton_flux,
        calculate_neutron_flux,
        calculate_low_E_neutron_flux
    )
    from LSDatm import convert_xyz_to_pressure
    


def apply_LSD_scaling_routine(
    lat : float,
    lon : float,
    alt : float,
    stdatm : bool = False,
    age : float = 0,
    w : float = -1,
    nuclide : int = 10,
    consts : dict = parse_LSDconsts(),
    era40 : dict = parse_LSDERA40()
):
    
    # This function calculates Lifton, Sato, and Dunai time-dependent scaling factors 
    # for a given set of inputs
    # syntax : LSD(lat,lon,alt,atm,age,nuclide)

    # lat = sample latitude in deg N (negative values for S hemisphere)
    # lon = sample longitude in deg E (negative values for W longitudes, 
    #     or 0-360 degrees E) 
    # alt = sample altitude in m above sea level
    # atm = atmospheric model to use: 1 for U.S. Standard Atmosphere, 
    #     0 for ERA-40 Reanalysis
    # age = age of sample
    # w = gravimetric fractional water content - 0.066 is default 
    #     typically about 14# volumetric per Fred Phillips. -1 gives default
    #     value
    # nuclide = nuclide of interest: 26 for 26Al, 10 for 10Be, 14 for 14C,
    #     3 for 3He, 0 for nucleon flux
    # 
    # Input values as scalars
    #
    # Based on code written by Greg Balco -- Berkeley
    # Geochronology Center
    # balcs@bgc.org
    # 
    # Modified by Brent Goehring and 
    # Nat Lifton -- Purdue University
    # nlifton@purdue.edu, bgoehrin@purdue.edu


    # Copyright 2013, Berkeley Geochronology Center and
    # Purdue University
    # All rights reserved
    # Developed in part with funding from the National Science Foundation.
    #
    # This program is free software you can redistribute it and/or modify
    # it under the terms of the GNU General Public License, version 3,
    # as published by the Free Software Foundation (www.fsf.org).


    # what version is this?
    ver = '1.0'

    is14 = 0
    is10 = 0
    is26 = 0
    is3 = 0
    isflux = 0

    # Load the input data structure
    sample = {
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "stdatm": stdatm,
        "age": age,
        "nuclide": nuclide
    }
    
    if nuclide == 14:
        is14 = 1
    elif nuclide == 10:
        is10 = 1
    elif nuclide == 26:
        is26 = 1    
    elif nuclide == 3:
        is3 = 1  
    else:
        isflux = 1
        
    # Make the time vector
    calFlag = 0

    # Age Relative to t0=2010
    tv = np.concatenate(
        (
            np.arange(0, 60, 10),
            np.arange(60, 50160, 100),
            np.arange(51060, 2001060, 1000),
            np.logspace(np.log10(2001060), 7, 200)
        )
    )
    LSDRc = np.zeros(tv.shape)

    # Need solar modulation parameter
    this_SPhi = np.zeros(tv.shape) + consts["SPhiInf"] # Solar modulation potential for Sato et al. (2008)
    this_SPhi[0:120] = consts["SPhi"] # Solar modulation potential for Sato et al. (2008)

    if w < 0:
        w = 0.066 # default gravimetric water content for Sato et al. (2008)
        
    # interpolate an M for tv > 7000...
    m_interpolator = sp.interpolate.interp1d(
        consts["t_M"], consts["M"]
    )
    temp_M = m_interpolator(tv[76:])

    # Pressure correction
    if sample["stdatm"] == True:
        gmr = -0.03417 # Assorted constants
        dtdz = 0.0065 # Lapse rate from standard atmosphere
        # Calculate site pressure using the Standard Atmosphere parameters with the
        # standard atmosphere equation.
        sample["pressure"] = 1013.25 * np.exp( (gmr/dtdz) * ( np.log(288.15) - np.log(288.15 - (alt*dtdz)) ) )
    else:
        # Use era40 reanalysis data
        sample["pressure"] = convert_xyz_to_pressure(sample["lat"],sample["lon"],sample["alt"],era40)

    # catch for negative longitudes before Rc interpolation
    if np.any(sample["lon"] < 0): 
        sample["lon"] = sample["lon"] + 360

    # Make up the Rc vectors.

    # Modified to work with new interpolation routines in MATLAB 2012a and later. 09/12
    loni, lati, tvi = np.meshgrid(sample["lon"],sample["lat"],tv[:76])
    
    rc_interpolator = sp.interpolate.RegularGridInterpolator(
        (consts["lat_Rc"], consts["lon_Rc"], consts["t_Rc"]), 
        consts["TTRc"],
        method="linear",
        bounds_error=True
    )

    LSDRc[0:76] = rc_interpolator(
        (lati, loni, tvi)
    )

    # Fit to Trajectory-traced GAD dipole field as f(M/M0), as long-term average. 
    # 
    dd = np.array([6.89901,-103.241,522.061,-1152.15,1189.18,-448.004])

    LSDRc[76:] = temp_M*(dd[0]*np.cos(sample["lat"]*np.pi/180) + \
        dd[1]*(np.cos(sample["lat"]*np.pi/180))**2 + \
        dd[2]*(np.cos(sample["lat"]*np.pi/180))**3 + \
        dd[3]*(np.cos(sample["lat"]*np.pi/180))**4 + \
        dd[4]*(np.cos(sample["lat"]*np.pi/180))**5 + \
        dd[5]*(np.cos(sample["lat"]*np.pi/180))**6) 

    # Next, chop off tv
    clipindex = np.where(tv <= sample["age"])[0][-1]
    tv2 = tv[:clipindex+1]
    if tv2[-1] < sample["age"]:
        tv2 = np.concatenate((tv2, np.array([sample["age"]])))

    # Now shorten the Rc's commensurately 
    intfun = sp.interpolate.interp1d(tv, LSDRc)
    LSDRc = intfun(tv2)
    intfun = sp.interpolate.interp1d(tv, this_SPhi)
    LSDSPhi = intfun(tv2)

    LSDout = _calculate_LSD_production_scaling(
        sample["pressure"],
        LSDRc,
        LSDSPhi,
        w,
        nuclide,
        consts
    )

    LSDout["tv"] = tv2
    LSDout["Rc"] = LSDRc
    LSDout["pressure"] = sample["pressure"]
    LSDout["alt"] = sample["alt"]

    # Write results to file

    return LSDout
    
def _calculate_LSD_production_scaling(
    h : np.ndarray,
    Rc : np.ndarray,
    SPhi,
    w : float,
    nuclide : int,
    consts = parse_LSDconsts()
):
    
    # Implements the Lifton Sato Dunai scaling scheme for spallation.
    #
    # Syntax: scalingfactor = LiftonSatoSX(h,Rc,SPhi,w,consts)
    #
    # Where:
    #   h = atmospheric pressure (hPa)
    #   Rc = cutoff rigidity (GV)
    #   SPhi = solar modulation potntial (Phi, see source paper)
    #   w = fractional water content of ground (nondimensional)
    #   
    #
    # Vectorized. Send in scalars or vectors of common length. 
    #

    # Written by Nat Lifton 2013, Purdue University
    # nlifton@purdue.edu
    # Based on code by Greg Balco -- Berkeley Geochronology Lab
    # balcs@bgc.org
    # April, 2007
    # Part of the CRONUS-Earth online calculators: 
    #      http://hess.ess.washington.edu/math
    #
    # Copyright 2001-2013, University of Washington, Purdue University
    # All rights reserved
    # Developed in part with funding from the National Science Foundation.
    #
    # This program is free software you can redistribute it and/or modify
    # it under the terms of the GNU General Public License, version 3,
    # as published by the Free Software Foundation (www.fsf.org).


    # what version is this?

    ver = '1.0'

    mfluxRef = consts["mfluxRef"]
    muRef = (mfluxRef["neg"] + mfluxRef["pos"])

    # Select reference values for nuclide of interest or flux

    if nuclide == 3:
        HeRef = consts["P3nRef"] + consts["P3pRef"]
    elif nuclide == 10:
        BeRef = consts["P10nRef"] + consts["P10pRef"]
    elif nuclide == 14:
        CRef = consts["P14nRef"] + consts["P14pRef"]
    elif nuclide == 26:
        AlRef = consts["P26nRef"] + consts["P26pRef"]
    else:
        SpRef = consts["nfluxRef"] + consts["pfluxRef"]
        # Sato et al. (2008) Reference hadron flux integral >1 MeV

    EthRef = consts["ethfluxRef"]
    ThRef = consts["thfluxRef"]

    # Site nucleon fluxes

    NSite = calculate_neutron_flux(h,Rc,SPhi,w,nuclide,consts)
    ethflux, thflux = calculate_low_E_neutron_flux(h,Rc,SPhi,w)
    PSite = calculate_proton_flux(h,Rc,SPhi,nuclide,consts)

    # Site omnidirectional muon flux
    mflux = calculate_muon_flux(h,Rc,SPhi) #Generates muon flux at site from Sato et al. (2008) model
    muSite = (mflux["neg"] + mflux["pos"])

    Site = {
        "muSF": np.zeros((len(Rc), len(mflux["E"])))
    }
    
    #Nuclide-specific scaling factors as f(Rc)
    if nuclide == 3:
        Site["He"] = (NSite["P3n"] + PSite["P3p"])/HeRef
    elif nuclide == 10:
        Site["Be"] = (NSite["P10n"] + PSite["P10p"])/BeRef
    elif nuclide == 14:
        Site["C"] = (NSite["P14n"] + PSite["P14p"])/CRef
    elif nuclide == 26:
        Site["Al"] = (NSite["P26n"] + PSite["P26p"])/AlRef
    else:    #Total nucleon flux scaling factors as f(Rc)
        Site["sp"] = ((NSite["nflux"] + PSite["pflux"]))/SpRef # Sato et al. (2008) Reference hadron flux integral >1 MeV

    Site["E"] = NSite["E"] #Nucleon flux energy bins
    Site["eth"] = ethflux/EthRef #Epithermal neutron flux scaling factor as f(Rc)
    Site["th"] = thflux/ThRef #Thermal neutron flux scaling factor as f(Rc)

    #Differential muon flux scaling factors as f(Energy, Rc)
    Site["muE"] = mflux["E"] #Muon flux energy bins (in MeV)
    Site["mup"] = mflux["p"] #Muon flux momentum bins (in MeV/c)

    for i in range(0, len(Rc)):
        Site["muSF"][i,:] = muSite[i,:]/muRef
    #Integral muon flux scaling factors as f(Rc)
    Site["muTotal"] = mflux["total"]/mfluxRef["total"] #Integral total muon flux scaling factor
    Site["mn"] = mflux["nint"]/mfluxRef["nint"] #Integral neg muon flux scaling factor
    Site["mp"] = mflux["pint"]/mfluxRef["pint"] #Integral pos muon flux scaling factor
    Site["mnabs"] = mflux["nint"] #Integral neg muon flux
    Site["mpabs"] = mflux["pint"] #Integral pos muon flux 
        
    return Site

if __name__ == "__main__":
    
    import time 
    
    consts = parse_LSDconsts()
    era40 = parse_LSDERA40()    
    
    for k in era40.keys():
        print(k, era40[k].shape)
    
    start = time.time()
    n_repeats = 1
    for i in range(0, n_repeats):
        output = apply_LSD_scaling_routine(
            lat=45,
            lon=45,
            alt=4000,
            stdatm=False,
            age=10,
            w=-1,
            nuclide=10,
            #consts=consts,
            #era40=era40
        )
    end = time.time()
    print(end-start)
    
    print("Spallogenic neutron scaling factors:", output["Be"])
    print("Epithermal neutron flux scaling factor:", output["eth"])
    print("Thermal neutron flux scaling factor:", output["th"])
    #print("Energy-dependent muon scaling factors:", output["muSF"])
    print("Total muon flux scaling factor:", output["muTotal"])
    print("Negative muon flux scaling factors:", output["mn"])
    print("Positive muon flux scaling factors:", output["mp"])

    