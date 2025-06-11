# pyLSD

__pyLSD__ is a simple translation of the Lifton-Sota-Dunai scaling scheme from the original Matlab source code into Python.
The original code used for this Python implementation can be found in the supplement to [Lifton et al. (2014)](https://doi.org/10.1016/j.epsl.2013.10.052).

# Usage

Using pyLSD is equivalent to using the Matlab version, altough some functions have been renamed to improve readibility. The main function to calculate scaling
factors is `apply_LSD_scaling_routine()`. It takes a __a single sample point__ as an input, this means that it has to be used in a loop when calculating scaling factors for different pixels in a
basin.

```
out = apply_LSD_scaling_routine(
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

The output us a dictionary containing various information (refer to the supplement of Lifton et al.) for more details. Importanty, the scaling factors for spallogenic neutron, epithermal neutron, thermal neutron, total muon, negative muon, and positive muon scaling factors can be accessed via:

```
print("Spallogenic neutron scaling factors:", output["Be"])
print("Epithermal neutron flux scaling factor:", output["eth"])
print("Thermal neutron flux scaling factor:", output["th"])
print("Total muon flux scaling factor:", output["muTotal"])
print("Negative muon flux scaling factors:", output["mn"])
print("Positive muon flux scaling factors:", output["mp"])
```

