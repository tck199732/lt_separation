from matplotlib.colors import ListedColormap
from typing import Dict


def set_mpl_style(mpl, config: Dict = {}):
    config_ = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "figure.dpi": 120,
        "figure.figsize": (4, 3.5),
        "figure.facecolor": "white",
        "xtick.top": True,
        "xtick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.right": True,
        "ytick.direction": "in",
        "ytick.minor.visible": True,
    }
    config_.update(config)

    for k, v in config_.items():
        mpl.rcParams[k] = v


def define_cmap(
    name: str,
    colors: list[str],
) -> ListedColormap:
    """
    Define a custom colormap.

    Parameters
    ----------
    name : str
        Name of the colormap.
    colors : list[str]
        List of color hex strings.

    Returns
    -------
    ListedColormap
        The defined colormap.
    """
    cmap = ListedColormap(colors, name=name)
    cmap.set_bad("white", 1.0)
    cmap.set_under("white", 1.0)
    return cmap
