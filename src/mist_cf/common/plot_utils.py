""" plot_utils.py 
Set plot utils
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

color_scheme = {
    "sirius":"#EFC7B8",# "#9970c1",
    "ffn_ms2": "#cc545e",
    "mist_cf_ms2": "#6F84AE", #"#568C63", 
    "backup": "#64a860",
    "ffn_ms1": "#b98d3e",
    "xformer": "#568C63", #"#638ccc",
}

# Define common matplotlib markers for each method for publication
marker_scheme = {
    "sirius": "o",
    "ffn_ms2": "s",
    "mist_cf_ms2": "D",
    "backup": "P",
    "ffn_ms1": "X",
    "xformer": "v",
}


model_order = ["ffn_ms1", "ffn_ms2", "xformer", 
               "mist_cf_ms2"]
sirius_order = ["sirius", "sirius_struct", "sirius_submit", "mist_cf_ms2", "mist_cf_ms2_50_peaks"]

rename_scheme = {
    "sirius": "SIRIUS",
    "ffn_ms2": "FFN",
    "mist_cf_ms2": "MIST-CF",
    "mist_cf_ms2_50_peaks": "MIST-CF (50 peaks)",
    "ffn_ms1": "MS1 Only",
    "xformer": "Transformer",
    "sirius_struct": "SIRIUS (CSI:FingerID)",
    "sirius_submit": "SIRIUS (Curated)",
}


ion_colors = [
    "#83c7bb",
    "#a0ead9",
    "#e7aed1",
    "#bed7a0",
    "#aebaeb",
    "#e5bb99",
    "#71cdeb",
]


mr_colors = [
    "#e6afd3",
    "#addcaf",
    "#74aff3",
    "#e8dca9",
    "#b4bcec",
    "#b9bd88",
    "#71cdeb",
    "#e8b098",
    "#8adbd3",
]


def cm2inch(value):
    return value / 2.54

def set_style():
    """set_style"""
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    sns.set(context="paper", style="ticks")
    mpl.rcParams["text.color"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["xtick.color"] = "black"
    mpl.rcParams["ytick.color"] = "black"
    mpl.rcParams["xtick.major.size"] = 2.5
    mpl.rcParams["ytick.major.size"] = 2.5
    mpl.rcParams["xtick.major.width"] = 0.45
    mpl.rcParams["ytick.major.width"] = 0.45
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.linewidth"] = 0.45
    mpl.rcParams["font.size"] = 9
    mpl.rcParams["axes.labelsize"] = 9
    mpl.rcParams["axes.titlesize"] = 9
    mpl.rcParams["figure.titlesize"] = 9
    mpl.rcParams["figure.titlesize"] = 9
    mpl.rcParams["legend.fontsize"] = 6
    mpl.rcParams["legend.title_fontsize"] = 9
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6


def set_size(w, h, ax=None):
    """w, h: width, height in inches
    ​
        Resize the axis to have exactly these dimensions

    """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


