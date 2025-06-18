import scipy as sp
import numpy as np
import os 

def _parse_file(
    fname : str
) -> dict:
    
    """
    Internal functions to deal with the .mat files of the original
    LSD Matlab code. This will parse them into a Python dict compatible
    with the other functions in this code.
    
    Written by Lennart Grimm (2025).
    
    Parameters:
    -----------
        fname : str
            The file name of the .mat file. This function expects it to be
            in the same dir as this Python file (LSDparse.py).
    
    Returns:
    --------
        out : dict
            The parsed data as a Python dict.
    """
    wd = os.path.dirname(os.path.realpath(__file__))
    mat_data = sp.io.loadmat(os.path.join(wd, fname))
    
    out = {}
            
    for k in mat_data.keys():
        if "__" not in k:
            if len(mat_data[k].shape) == 2:
                # if one of the dimensions is just 1, cast to 1d
                if 1 in mat_data[k].shape:
                    out[k] = np.squeeze(mat_data[k])
                else:
                    out[k] = mat_data[k]
            elif len(mat_data[k].shape) == 1:
                out[k] = mat_data[k][:,0]
            else: # 3D or any other case
                out[k] == mat_data[k]
            
    return out

def parse_LSDXSectsReedyAll() -> dict:
    """
    Lazy wrapper around _parse_file() for .mat files.
    
    Written by Lennart Grimm (2025).
    """
    return _parse_file("LSDXSectsReedyAll.mat")

def parse_LSDERA40() -> dict:
    """
    Lazy wrapper around _parse_file() for .mat files.
    
    Written by Lennart Grimm (2025).
    """
    return _parse_file("LSDERA40.mat")

def parse_LSDPMag_Sep12() -> dict:
    """
    Lazy wrapper around _parse_file() for .mat files.
    
    Written by Lennart Grimm (2025).
    """
    return _parse_file("LSDPMag_Sep12.mat")

def parse_LSDReference() -> dict:
    """
    Lazy wrapper around _parse_file() for .mat files.
    
    Written by Lennart Grimm (2025).
    """
    return _parse_file("LSDReference.mat")

def parse_LSDconsts() -> dict:

    """
    Function to parse LSDconsts. This structure is a bit more complex, since the
    last argument is another nested structure. This function attempts to parse it
    explicitly.
    
    Written by Lennart Grimm (2025).
    
    Parameters:
    -----------
    
    Returns:
    --------
        out : dict
            The parsed data.
    """
    
    fieldnames = [
        'version',
        'prepdate',
        'Natoms3',
        'Natoms10',
        'Natoms14',
        'Natoms26',
        'k_neg10',
        'delk_neg10',
        'k_neg14',
        'delk_neg14',
        'k_neg26',
        'delk_neg26',
        'sigma190_10'   ,
        'delsigma190_10',
        'sigma190_14'   ,
        'delsigma190_14',
        'sigma190_26'   ,
        'delsigma190_26',
        'O16nxBe10' ,
        'O16pxBe10' ,
        'SinxBe10',
        'SipxBe10',
        'O16nn2pC14',
        'O16pxC14',
        'SinxC14',
        'SipxC14',
        'Aln2nAl26' ,
        'AlppnAl26' ,
        'SinxAl26',
        'SipxAl26',
        'KnxCl36',
        'KpxCl36',
        'CanapCl36' ,
        'CapxCl36',
        'FenxCl36',
        'FepxCl36',
        'TinxCl36',
        'TipxCl36',
        'MgnxNe21',
        'MgpxNe21',
        'AlnxNe21',
        'AlpxNe21',
        'SinxNe21',
        'SipxNe21',
        'OnxHe3T',
        'OpxHe3T',
        'SinxHe3T',
        'SipxHe3T',
        'AlnxHe3T',
        'AlpxHe3T',
        'MgnxHe3T',
        'MgpxHe3T',
        'CanxHe3T',
        'CapxHe3T',
        'FenxHe3T',
        'FepxHe3T',
        'M',
        't_M',
        't_fineRc',
        'TTRc'   ,
        'IHRc'   ,
        'lat_Rc' ,
        'lon_Rc' ,
        't_Rc'   ,
        'SPhi'   ,
        'SPhiInf',
        'E',
        'P3nRef',
        'P3pRef' ,
        'P10nRef',
        'P10pRef',
        'P14nRef',
        'P14pRef',
        'P26nRef',
        'P26pRef',
        'nfluxRef',
        'pfluxRef',
        'ethfluxRef',
        'thfluxRef' ,
        'mfluxRef'
    ]
    
    mflux_Ref_fieldnames = [
        'total',
        'neg'  ,
        'pos'  ,
        'nint' ,
        'pint' ,
        'E'    ,
        'p'    
    ]
    
    wd = os.path.dirname(os.path.realpath(__file__))
    mat_data = sp.io.loadmat(os.path.join(wd, "LSDconsts.mat"))
    consts = mat_data["consts"][0,0]
    
    out = {}
    for i, name in enumerate(fieldnames[:-1]):
        if len(consts[i].shape) == 2:
            out[name] = consts[i][0,:]
        elif len(consts[i].shape) == 1:
            out[name] = consts[i][0]
        else: 
            out[name] = consts[i]
    
    # treat the last entry separately, because it is another struct...
    nested_struct = consts[-1][0,0]
    nested_out = {}
    
    # Slightly verbose, but deals with dimensions of arrays...
    for i, name in enumerate(mflux_Ref_fieldnames):
        if len(nested_struct[i].shape) == 2:
            if nested_struct[i].shape == (1,1):
                nested_out[name] = nested_struct[i][0,0]
            else:
                nested_out[name] = nested_struct[i][0,:]
        elif len(nested_struct[i].shape) == 1:
            nested_out[name] = nested_struct[i][0]
        else:
            nested_out[name] = nested_struct[i]
    out[fieldnames[-1]] = nested_out
    
    return out

if __name__ == "__main__":
    consts = parse_LSDconsts()
    xsectsreedyall = parse_LSDXSectsReedyAll()
    era40 = parse_LSDERA40()
    pmagsep12 = parse_LSDPMag_Sep12()
    reference = parse_LSDReference()

    
