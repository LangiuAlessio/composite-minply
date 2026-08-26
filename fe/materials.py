"""Orthotropic lamina registry. Moduli in MPa, density in tonne/mm^3.

Moduli from the WWFE dataset: Soden, Hinton & Kaddour, Compos. Sci. Technol. 58
(1998) 1011-1022 (NPTEL Table 3.1). GFRP G23 from transverse isotropy
G23 = E2 / (2*(1+nu23)); GFRP density 2.0e-9 tonne/mm^3 (typical glass/epoxy).
The T300/epoxy entry is a generic CFRP lamina used by fe/plate_model.py and by the
closed-form canary of exp9. It is NOT the lamina of the design campaign: that one is
canale2018 below, whose values are those printed in the paper (Table: Material
properties) and are the ones optimisers/constrained_search.py writes into its decks.
The two differ (E1 135000 vs 125100 MPa, ply 0.125 vs 0.1 mm) and must not be swapped.
"""

_WWFE = "Soden, Hinton & Kaddour, Compos. Sci. Technol. 58 (1998) 1011-1022 (WWFE)"

MATERIALS: dict[str, dict] = {
    "T300/epoxy": dict(
        E1=135000.0, E2=10000.0, E3=10000.0,
        nu12=0.3, nu13=0.3, nu23=0.4,
        G12=5000.0, G13=5000.0, G23=3500.0,
        rho=1.6e-9,
        source="Typical T300/epoxy CFRP (consistent with " + _WWFE + " T300/BSL914C)",
    ),
    # Design-campaign lamina: in-plane constants from Canale et al. (2018), out-of-plane
    # constants their transversely-isotropic completion (that work does not report them).
    # These are the values printed in the paper's material table and the ones written into
    # the CalculiX decks by optimisers/constrained_search.py. Ply thickness 0.1 mm.
    "canale2018": dict(
        E1=125100.0, E2=7840.0, E3=7840.0,
        nu12=0.3, nu13=0.15, nu23=0.15,
        G12=4600.0, G13=4000.0, G23=4000.0,
        rho=1.62e-9,
        source="Canale et al. (2018) in-plane constants; out-of-plane by transverse isotropy",
    ),
    "E-glass/epoxy": dict(
        E1=53480.0, E2=17700.0, E3=17700.0,
        nu12=0.278, nu13=0.278, nu23=0.4,
        G12=5830.0, G13=5830.0, G23=17700.0 / (2 * (1 + 0.4)),
        rho=2.0e-9,
        source=_WWFE + " E-glass/LY556/HT907/DY063; rho typ. glass/epoxy 2.0 g/cm^3",
    ),
    "S-glass/epoxy": dict(
        E1=45600.0, E2=16200.0, E3=16200.0,
        nu12=0.278, nu13=0.278, nu23=0.4,
        G12=5830.0, G13=5830.0, G23=16200.0 / (2 * (1 + 0.4)),
        rho=2.0e-9,
        source=_WWFE + " S-glass/MY750/HY917/DY063; rho typ. glass/epoxy 2.0 g/cm^3",
    ),
}


def get_material(name: str) -> dict:
    """Return the lamina property dict for `name`.

    Tolerant of common aliases an LLM/user may type: exact key, case-insensitive,
    or the fibre prefix before '/' (e.g. 'T300' -> 'T300/epoxy', 'e-glass' ->
    'E-glass/epoxy'). Unknown names raise KeyError listing the valid keys."""
    if name in MATERIALS:
        return MATERIALS[name]
    low = name.strip().lower()
    for key, props in MATERIALS.items():
        kl = key.lower()
        if low == kl or low == kl.split("/")[0]:
            return props
    raise KeyError(f"Unknown material {name!r}. Valid: {sorted(MATERIALS)}")
