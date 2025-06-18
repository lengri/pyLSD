import numpy as np
import scipy as sp

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
    age : float = 0.,
    w : float = -1.,
    nuclide : int = 10,
    consts : dict = parse_LSDconsts(),
    era40 : dict = parse_LSDERA40()
):
    
    """
    Apply the LSD scaling scheme to a single input point, specified by latitue (decimal degrees), 
    longitude (decimal degrees), and elevation (m). The sample can also have a non-zero age, e.g.
    if it was buried.
    
    Original Matlab code written by Greg Balco (2007), modified by Nat Lifton (2011), translated
    to Python by Lennart Grimm (2025).
    
    Parameters:
    -----------
        lat : float
            Sample point latitude in decimal degrees.
        lon : float
            Sample point longitude in decimal degrees.
        alt : float
            Sample point elevation in meters.
        stdatm : bool
            If True, use the standard atmosphere equation instead of EAR40 data to derive site pressures.
        age : float
            Sample age in years.
        w : float
            Gravimetric fractional water content. Leave at -1 to set the default value.
        nuclide : int
            Mass number of the nuclide of interest. Available options are 3 - He, 10 - Be,
            14 - C, 26 - Al.
        consts : dict
            All the constants going into the LSD scaling scheme. By default, a dict called from
            parse_LSDconsts().
        era40 : dict
            ERA40 reanalyis data used for calculating site-specific pressure if stdatm=False. By default,
            a dict called from parse_LSDERA40().
        
    Returns:
    --------
        Site : dict
            A dict containing the various scaling factors at the sample site for the specified nuclide.
            Most important are probably "Be" (spallation scaling, this will be the name of the input nuclide), 
            "eth" (epithermal neutron scaling), "th" (thermal neutrons scaling), "muSF" (energy dependent 
            muon scaling), "muTotal" (Total muon flux scaling), "mn" (negative muon scaling), "mp" (positive muon
            scaling).
            There is more data available in the dict, see also the original Matlab code for more information.
    """

    # what version is this?
    ver = '1.0'
    
    # Load the input data structure
    sample = {
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "stdatm": stdatm,
        "age": age,
        "nuclide": nuclide
    }

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
    h : float,
    Rc : np.ndarray,
    SPhi : float,
    w : float,
    nuclide : int,
    consts = parse_LSDconsts()
) -> dict:
    
    """
    Internal function to calculate scaling factors using pressure and cutoff rigidity.
    
    Original Matlab code written by Greg Balco (2007), modified by Nat Lifton (2013), translated
    to Python by Lennart Grimm (2025).
    
    Parameters:
    -----------
        h : float
            Site atmospheric pressure in hPa.
        Rc : np.ndarray
            Cutoff rigidities in GV.
        SPhi : float
            Solar modulation potential.
        w : float
            Gravimetric fractional water content. Leave at -1 to set the default value.
        nuclide : int
            Mass number of the nuclide of interest. Available options are 3 - He, 10 - Be,
            14 - C, 26 - Al.
        consts : dict
            All the constants going into the LSD scaling scheme. By default, a dict called from
            parse_LSDconsts().
    
    Returns:
    --------
        out : dict
            A dict of scaling factors.
    """

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

    