import json
import logging
import pathlib
import random
import string
import importlib
import hashlib
import numpy as np
from typing import Callable, List, Any, Tuple, Optional
from collections.abc import Iterable
from collections import OrderedDict


def load_json(fname: str | pathlib.Path) -> OrderedDict:
    """
    Load JSON file and return as an OrderedDict.
    fname : str | pathlib.Path
        Path to the JSON file.
    Returns
    -------
    OrderedDict
        Contents of the JSON file as an OrderedDict.
    """
    fname = pathlib.Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: dict, fname: str | pathlib.Path):
    """
    Write a dictionary to a JSON file.
    content : dict
        Dictionary to write to the JSON file.
    fname : str | pathlib.Path
        Path to the JSON file.
    """
    fname = pathlib.Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def map_dict(func: Callable, input_dict: dict, inplace: bool = False) -> dict | None:
    """
    Apply a function to all values in the inner-most level of a dictionary, recursively.
    Parameters
    ----------
    func : Callable
        Function to apply to each value.
    input_dict : dict
        Input dictionary.
    inplace : bool, optional
        If True, modify the input dictionary in place. If False, return a new dictionary. Defaults to False.
    Returns
    -------
    dict | None
        New dictionary with the function applied to each value, or None if inplace is True.
    """

    out_dict = input_dict if inplace else input_dict.copy()
    for key, value in input_dict.items():
        if isinstance(value, dict):
            out_dict[key] = map_dict(func, value)
        elif isinstance(value, (list, tuple)):
            out_dict[key] = type(value)(map_dict(func, v) for v in value)
        else:
            out_dict[key] = func(value)

    if not inplace:
        return out_dict


def get_random_str(length: int = 10) -> str:
    """
    Generate a random string of fixed length. The string consists of uppercase and lowercase letters.
    Parameters
    ----------
    length : int, optional
        Length of the random string. Defaults to 10.
    Returns
    -------
    str
        Randomly generated string.
    """
    letters = string.ascii_letters
    return "".join(random.choice(letters) for _ in range(length))


def hash_from_str(code: str, prefix: Optional[str] = None, digest_size: int = 4) -> str:
    """
    Generate a short deterministic hash for a given string

    Parameters
    ----------
    code : str
        The input string to hash.
    prefix : str, optional
        Optional prefix for the function name, e.g. 'f' -> f_<hash>. Defaults to None.
    digest_size : int, optional
        Size of the hash digest in bytes. Defaults to 4.

    Returns
    -------
    str
        A short unique name like 'f_a1b2c3d4'.
    """
    h = hashlib.blake2b(code.encode("utf-8"), digest_size=digest_size).hexdigest()
    return f"{prefix}_{h}" if prefix is not None else h


def filter_existing_paths(
    struct: Iterable | pathlib.Path | str,
) -> Iterable | pathlib.Path | str:
    """
    Recursively filter a nested structure of pathlib.Path objects (inside lists/tuples)
    so that only existing paths are kept. Preserves the nested structure.
    Parameters
    ----------
    struct : Iterable | pathlib.Path | str
        A nested structure (lists/tuples) of pathlib.Path objects or strings representing paths.
    Returns
    -------
    Iterable | pathlib.Path | str
        The same nested structure but with non-existing paths removed.
    """
    if isinstance(struct, pathlib.Path):
        return struct if struct.exists() else None

    if isinstance(struct, (list, tuple)):
        filtered = [filter_existing_paths(item) for item in struct]
        filtered = [item for item in filtered if item is not None]
        return type(struct)(filtered)  # preserve list/tuple type

    return struct


def flatten_paths(struct: Iterable | pathlib.Path | str) -> list[pathlib.Path]:
    """
    Recursively flatten a nested structure of pathlib.Path into a list.
    Parameters
    ----------
    struct : Iterable | pathlib.Path | str
        A nested structure (lists/tuples) of pathlib.Path objects or strings representing paths.
    Returns
    -------
    list[pathlib.Path]
        A flat list of pathlib.Path objects.
    """
    if isinstance(struct, pathlib.Path):
        return [struct]
    elif isinstance(struct, str):
        return [pathlib.Path(struct)]
    elif isinstance(struct, Iterable) and not isinstance(struct, (str, bytes)):
        paths = []
        for item in struct:
            paths.extend(flatten_paths(item))
        return paths
    else:
        return []


def import_attr(attr_name: str, avail_modules: List[str] | str) -> Tuple[Any, str]:
    """
    Dynamically import an attribute (class, function, variable) from a list of available modules. By default, returns the first successfully imported attribute. If the attribute is not found in any of the provided modules, raises an ImportError with detailed information about the attempts.
    Parameters
    ----------
    attr_name : str
        Name of the attribute to import.
    avail_modules : List[str] | str
        List of module names (as strings) to search for the attribute.
    Returns
    -------
    Tuple[Any, str]
        The imported attribute and module name.
    Raises
    ------
    ImportError
        If the attribute is not found in any of the provided modules.
    Example
    -------
    attr, mod_name = import_attr("Path", ["os", "pathlib"])
    assert attr.__name__ == "Path"
    assert mod_name == "pathlib"
    """

    if isinstance(avail_modules, str):
        avail_modules = [avail_modules]

    if not avail_modules:
        raise ValueError("No modules provided in `avail_modules`.")

    log_msg = {mod: [] for mod in avail_modules}

    for mod_name in avail_modules:
        try:
            module = importlib.import_module(mod_name)
        except ImportError as e:
            log_msg[mod_name].append(str(e))
            continue

        attr = getattr(module, attr_name, None)
        if attr is not None:
            return attr, mod_name
        else:
            log_msg[mod_name].append("Attribute not found.")

    available_str = ", ".join(avail_modules)
    log_details = "\n".join(
        [f"In module '{mod}': {', '.join(errors)}" for mod, errors in log_msg.items()]
    )
    raise ImportError(
        f"Could not find attribute '{attr_name}' in any of: [{available_str}].\n"
        f"Details of import attempts:\n{log_details}"
    )


def get_logger(
    name: str,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    file: str | None = None,
):
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        filename=file,
        filemode="a" if file else None,
    )
    logger = logging.getLogger(name)
    return logger
