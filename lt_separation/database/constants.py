import warnings


try:
    import ROOT

    pdg = ROOT.TDatabasePDG.Instance()
except ImportError:
    pdg = None
    warnings.warn("ROOT is not imported. Falling back to default masses.")

DEFAULT_MASSES = {
    211: 0.13957,
    321: 0.493677,
    2212: 0.9382720813,
    3122: 1.115683,
}


def get_mass(pdg_code):
    if pdg is None:
        try:
            return DEFAULT_MASSES[pdg_code]
        except KeyError:
            raise ValueError(f"Mass for PDG code {pdg_code} is not available.")
    return pdg.GetParticle(pdg_code).Mass()


MASS_PION_PLUS = get_mass(211)  # pi+
MASS_KAON_PLUS = get_mass(321)  # K+
MASS_PROTON = get_mass(2212)  # p
MASS_LAMBDA0 = get_mass(3122)  # Lambda^0
