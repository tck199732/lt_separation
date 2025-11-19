import sys
import gc
import subprocess
import pathlib
import pandas as pd
import numpy as np
from typing import Optional


try:
    import ROOT
except:
    path_root = (
        subprocess.check_output("root-config --libdir", shell=True)
        .decode("utf-8")
        .strip()
    )
    path_root = pathlib.Path(path_root)
    if not path_root.exists():
        raise FileNotFoundError(f"ROOT library path {path_root} does not exist.")
    sys.path.append(str(path_root))
    import ROOT


def load_RDataFrame(
    fnames: str | pathlib.Path | list[str | pathlib.Path],
    tr_name: str,
    set_alias: bool = True,
    implicit_mt: bool = True,
    n_threads: Optional[int] = None,
):
    """
    A wrapper to load ROOT RDataFrame from files.
    Parameters
    ----------
    fnames : str | pathlib.Path | list[str | pathlib.Path]
        A single filename or a list of filenames.
    tr_name : str
        Name of the tree to load.
    set_alias : bool, optional
        Whether to set aliases for columns with '.' in their names. Default is True.
    implicit_mt : bool, optional
        Whether to enable implicit multi-threading. Default is True.
    n_threads : Optional[int], optional
        Number of threads to use for implicit multi-threading. If None, use all available cores. Default is None.
    Returns
    -------
    ROOT.RDataFrame
        The loaded RDataFrame.
    """
    if implicit_mt and n_threads is None:
        ROOT.EnableImplicitMT()
    elif implicit_mt and n_threads is not None:
        ROOT.EnableImplicitMT(n_threads)
    else:
        ROOT.DisableImplicitMT()

    if isinstance(fnames, str) or isinstance(fnames, pathlib.Path):
        fnames = [fnames]

    fnames = list(map(str, fnames))
    rdf = ROOT.RDataFrame(tr_name, fnames)

    if set_alias:
        aliases = {}
        for col in rdf.GetColumnNames():
            if not "." in col:
                continue
            alias = col.replace(".", "_")
            if not alias in rdf.GetColumnNames() and not alias in aliases:
                rdf = rdf.Alias(alias, col)
                aliases[alias] = col
    return rdf


def TH1_to_numpy(histo):
    """
    Convert a ROOT TH1D or TH2D to numpy arrays.
    Parameters
    ----------
    histo : ROOT.TH1D or ROOT.TH2D
        The input ROOT histogram.
    Returns
    -------
    tuple of np.ndarray
        For 1D histogram: (x_centers, y_values, y_errors)
        For 2D histogram: (x_centers, y_centers, z_values, z_errors)
    """

    # if it is RDF result, dereference the pointer
    if hasattr(histo, "GetValue") and not isinstance(histo, ROOT.TH1):
        histo = histo.GetValue()

    if isinstance(histo, ROOT.TH1):
        if isinstance(histo, ROOT.TH2):
            x = np.array(
                [
                    histo.GetXaxis().GetBinCenter(b)
                    for b in range(1, histo.GetNbinsX() + 1)
                ]
            )
            y = np.array(
                [
                    histo.GetYaxis().GetBinCenter(b)
                    for b in range(1, histo.GetNbinsY() + 1)
                ]
            )
            content = np.array(
                [histo.GetBinContent(b) for b in range((len(x) + 2) * (len(y) + 2))]
            )
            content = content.reshape(len(x) + 2, len(y) + 2, order="F")
            content = content[1:-1, 1:-1]

            error = np.array(
                [histo.GetBinError(b) for b in range((len(x) + 2) * (len(y) + 2))]
            )
            error = error.reshape(len(x) + 2, len(y) + 2, order="F")
            error = error[1:-1, 1:-1]

            result = (x, y, content, error)
        else:
            x = [
                histo.GetXaxis().GetBinCenter(b)
                for b in range(1, histo.GetNbinsX() + 1)
            ]
            y = [histo.GetBinContent(b) for b in range(1, histo.GetNbinsX() + 1)]
            yerr = [histo.GetBinError(b) for b in range(1, histo.GetNbinsX() + 1)]
            result = (np.array(x), np.array(y), np.array(yerr))

    else:
        raise TypeError(f"Input must be a ROOT.TH1 or ROOT.TH2, got {type(histo)}")

    return result


def numpy_to_TH1(*args, name: str, title: str = ""):
    """
    Convert numpy arrays to ROOT TH1D or TH2D. Only uniform binning is supported.
    Parameters
    ----------
    *args : tuple of np.ndarray
        For 1D histogram: (x_centers, y_values) or (x_centers, y_values, y_errors)
        For 2D histogram: (x_centers, y_centers, z_values) or (x_centers, y_centers, z_values, z_errors)
    name : str
        Name of the histogram.
    title : str, optional
        Title of the histogram. Default is an empty string.
    Returns
    -------
    ROOT.TH1D or ROOT.TH2D
        The resulting ROOT histogram.
    """

    if len(args) in (2, 3) and args[-1].ndim == 1:
        nbins, dx = len(args[0]), np.diff(args[0]).mean()
        histo = ROOT.TH1D(name, title, nbins, args[0][0] - dx / 2, args[0][-1] + dx / 2)
        for i in range(nbins):
            histo.SetBinContent(i + 1, args[1][i])
            if len(args) == 3:
                histo.SetBinError(i + 1, args[2][i])
            else:
                histo.SetBinError(i + 1, 0)

    elif len(args) in (3, 4) and args[-1].ndim == 2:

        nbin_x, dx = len(args[0]), np.diff(args[0]).mean()
        nbin_y, dy = len(args[1]), np.diff(args[1]).mean()
        histo = ROOT.TH2D(
            name,
            title,
            nbin_x,
            args[0][0] - dx / 2,
            args[0][-1] + dx / 2,
            nbin_y,
            args[1][0] - dy / 2,
            args[1][-1] + dy / 2,
        )
        for i in range(nbin_x):
            for j in range(nbin_y):
                histo.SetBinContent(i + 1, j + 1, args[2][i, j])
                if len(args) == 4:
                    histo.SetBinError(i + 1, j + 1, args[3][i, j])
                else:
                    histo.SetBinError(i + 1, j + 1, 0)
    else:
        raise ValueError("Input numpy array has wrong shape.")

    return histo


def df_to_TH1D(
    df: pd.DataFrame,
    hname: str,
    htitle: str = "",
    xname: str = "x",
    yname: str = "y",
    yerr_name=None,
):
    """
    Convert a pandas DataFrame to a ROOT TH1D.
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data.
    hname : str
        Name of the histogram.
    htitle : str, optional
        Title of the histogram. Default is an empty string.
    xname : str, optional
        Name of the column to use for x values. Default is "x".
    yname : str, optional
        Name of the column to use for y values. Default is "y".
    yerr_name : str, optional
        Name of the column to use for y errors. If None, no errors are set.
    Returns
    -------
    ROOT.TH1D
        The resulting ROOT TH1D histogram.
    """

    assert xname in df.columns, f"{xname} not in dataframe columns"
    assert yname in df.columns, f"{yname} not in dataframe columns"
    if yerr_name is not None:
        assert yerr_name in df.columns, f"{yerr_name} not in dataframe columns"

    nbins, dx = len(df), df.x.values[1] - df.x.values[0]
    histo = ROOT.TH1D(
        hname, htitle, nbins, df.x.values[0] - dx / 2, df.x.values[-1] + dx / 2
    )

    assert np.allclose(
        [histo.GetXaxis().GetBinCenter(b) for b in range(1, histo.GetNbinsX() + 1)],
        df[xname].values,
    )

    for i in range(nbins):
        histo.SetBinContent(i + 1, df[yname].values[i])
        if yerr_name is not None:
            histo.SetBinError(i + 1, df[yerr_name].values[i])
        else:
            histo.SetBinError(i + 1, 0)
    return histo


def TH1D_to_df(histo, xname="x", yname="y"):
    """
    Convert a ROOT TH1D to a pandas DataFrame.
    Parameters
    ----------
    histo : ROOT.TH1D
        The input ROOT TH1D histogram.
    xname : str, optional
        Name of the x column in the DataFrame. Default is "x".
    yname : str, optional
        Name of the y column in the DataFrame. Default is "y".
    Returns
    -------
    pd.DataFrame
        The resulting DataFrame with columns for x, y, y error, and fractional y error
    """
    x = [histo.GetXaxis().GetBinCenter(b) for b in range(1, histo.GetNbinsX() + 1)]
    y = [histo.GetBinContent(b) for b in range(1, histo.GetNbinsX() + 1)]
    yerr = [histo.GetBinError(b) for b in range(1, histo.GetNbinsX() + 1)]
    yferr = np.divide(yerr, y, out=np.zeros_like(y), where=(np.array(y) != 0))
    return pd.DataFrame(
        {xname: x, yname: y, f"{yname}err": yerr, f"{yname}ferr": yferr}
    )


def TH2D_to_df(histo, xname="x", yname="y", zname="z", keep_zeros=True):
    """
    Convert a ROOT TH2D to a pandas DataFrame.
    Parameters
    ----------
    histo : ROOT.TH2D
        The input ROOT TH2D histogram.
    xname : str, optional
        Name of the x column in the DataFrame. Default is "x".
    yname : str, optional
        Name of the y column in the DataFrame. Default is "y".
    zname : str, optional
        Name of the z column in the DataFrame. Default is "z".
    keep_zeros : bool, optional
        Whether to keep entries with zero z values. Default is True.
    Returns
    -------
    pd.DataFrame
        The resulting DataFrame with columns for x, y, z, z error, and fractional z error
    """
    x = np.array(
        [histo.GetXaxis().GetBinCenter(b) for b in range(1, histo.GetNbinsX() + 1)]
    )
    y = np.array(
        [histo.GetYaxis().GetBinCenter(b) for b in range(1, histo.GetNbinsY() + 1)]
    )

    content = np.array(histo)
    content = content.reshape(len(x) + 2, len(y) + 2, order="F")
    content = content[1:-1, 1:-1]

    error = np.array([histo.GetBinError(b) for b in range((len(x) + 2) * (len(y) + 2))])
    error = error.reshape(len(x) + 2, len(y) + 2, order="F")
    error = error[1:-1, 1:-1]

    xx, yy = np.meshgrid(x, y, indexing="ij")
    df = pd.DataFrame(
        {
            xname: xx.flatten(),
            yname: yy.flatten(),
            zname: content.flatten(),
            f"{zname}err": error.flatten(),
        }
    )
    mask = df[zname] != 0.0
    df[f"{zname}_ferr"] = np.where(mask, np.abs(df[f"{zname}err"] / df[zname]), 0.0)
    return df if keep_zeros else df.query(f"{zname} != 0.0").reset_index(drop=True)


def get_root_keys(fname: str | pathlib.Path) -> list[str]:
    """
    Get the list of keys in a ROOT file.
    Parameters
    ----------
    fname : str | pathlib.Path
        Path to the ROOT file.
    Returns
    -------
    list[str]
        List of keys in the ROOT file.
    """
    f = ROOT.TFile(str(fname), "read")
    keys = [k.GetName() for k in f.GetListOfKeys()]
    f.Close()
    return keys


def get_root_obj(fname: str | pathlib.Path, histname: str):
    """
    Get a ROOT object (e.g., histogram) from a ROOT file.
    Parameters
    ----------
    fname : str | pathlib.Path
        Path to the ROOT file.
    histname : str
        Name of the histogram or object to retrieve.
    Returns
    -------
    ROOT.TObject
        The retrieved ROOT object.
    Raises
    -------
    ValueError
        If the histogram is not found in the file.
    """
    f = ROOT.TFile(str(fname), "read")
    keys = get_root_keys(fname)
    if not histname in keys:
        raise ValueError(f"Histogram {histname} not found in file {fname}.")
    h = f.Get(histname)
    h.SetDirectory(0)
    f.Close()
    return h


def save_root_obj(histo: ROOT.TObject, output_path: str | pathlib.Path):
    """
    Save ROOT histogram(s) to a ROOT file.
    Parameters
    ----------
    histo : ROOT.TObject
        A single ROOT histogram or a dictionary of histograms to
    output_path : str | pathlib.Path
        Path to the output ROOT file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outf = ROOT.TFile(str(output_path), "RECREATE")
    histo.Write()
    outf.Close()


def save_root_objs(histos: dict[str, ROOT.TObject], output_path: str | pathlib.Path):
    """
    Save ROOT histogram(s) to a ROOT file.
    Parameters
    ----------
    histos : dict[str, ROOT.TObject]
        A dictionary of histograms to save, with keys as histogram names.
    output_path : str | pathlib.Path
        Path to the output ROOT file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outf = ROOT.TFile(str(output_path), "RECREATE")
    for name, histo in histos.items():
        histo.SetName(name)
        histo.Write()
    outf.Close()
