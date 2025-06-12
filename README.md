# pyLSD

__pyLSD__ is a simple translation of the Lifton-Sato-Dunai scaling scheme from the Matlab source code into Python.
The original code that this Python implementation is based on can be found in the supplement to 
[Lifton et al. (2014)](https://doi.org/10.1016/j.epsl.2013.10.052).

# Installation

It is recommended to install pyLSD into a separate virtual environment to avoid any package version incompatibilities.

In your console, type 
```
python -m venv lsd
cd lsd\Scripts
activate
pip install git+https://github.com/lengri/pyLSD.git
```
Voilà! 

# Usage

Using pyLSD is mostly equivalent to using the Matlab version, altough some functions have been renamed to improve readibility. 
The main function to calculate scaling factors is `apply_LSD_scaling_routine()`. It takes a __a single sample point__ as an input, 
this means that it has to be used in a loop when calculating scaling factors for different pixels in a basin.

```
import pyLSD as lsd

out = lsd.apply_LSD_scaling_routine(
    lat = 45.,
    lon = 45.,
    alt = 2000.,
    stdatm = False,
    age = 0.,
    w = -1.,
    nuclide = 10,
    consts = parse_LSDconsts(),
    era40 = parse_LSDERA40()
)
```

The output is a dictionary containing various information (please refer to the supplement of [Lifton et al.](https://doi.org/10.1016/j.epsl.2013.10.052) for more details). 
Importanty, the scaling factors for spallogenic neutron, epithermal neutron, thermal neutron, total muon, negative muon, 
and positive muon production can be accessed via:

```
print("Spallogenic neutron scaling factors:", output["Be"])
print("Epithermal neutron flux scaling factor:", output["eth"])
print("Thermal neutron flux scaling factor:", output["th"])
print("Total muon flux scaling factor:", output["muTotal"])
print("Negative muon flux scaling factors:", output["mn"])
print("Positive muon flux scaling factors:", output["mp"])
```

# Notes

Site-specific pressure and surface temperature can be calculated in two ways: First, by using the standard atmosphere model 
([NOAA, 1976](https://www.ngdc.noaa.gov/stp/space-weather/online-publications/miscellaneous/us-standard-atmosphere-1976/us-standard-atmosphere_st76-1562_noaa.pdf)). 
To use this option, set `stdatm = True` when calling `lsd.apply_LSD_scaling_routine()`. Second, the default option with `stdatm=False`: 
Using ERA40 reanalysis data of pressure and temperature ([Dee et al., 2011](https://doi.org/10.1002/qj.828)). 
This should be the preferred option especially for sites outside North America.