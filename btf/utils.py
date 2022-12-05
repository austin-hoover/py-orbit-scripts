import numpy as np
import pandas as pd


def load_history(filename):
    df = pd.read_table(filename, sep=' ')
    # # Convert rms x and y emittance from m-rad to mm-mrad
    # df[['eps_x', 'eps_y']] *= 1e3 * 1e3  
    # # Convert rms z emittance from m-GeV to mm-keV
    # df[['eps_z']] *= 1e3 * 1e6
    return df


def load_bunch(filename, dims=None, dframe=False):
    names = ["x", "xp", "y", "yp", "z", "dE"]
    cols = list(range(6))
    if dims is not None:
        cols = [d if type(d) is int else names.index(d) for d in dims]
    names = [names[c] for c in cols]
    df = pd.read_table(filename, sep=' ', skiprows=14, usecols=cols, names=names)
    # Convert to mm, mrad, keV
    for col in ['x', 'xp', 'y', 'yp', 'z']:
        if col in df.columns:
            df[col] *= 1e3
    if 'dE' in df.columns:
        df['dE'] *= 1e6
    if dframe:
        return df
    return df.values