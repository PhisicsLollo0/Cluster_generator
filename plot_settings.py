import matplotlib.pyplot as plt

# Define LaTeX font settings
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10 * 2,
    "font.size": 10 * 2,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8 * 2,
    "xtick.labelsize": 8 * 2,
    "ytick.labelsize": 8 * 2,
}


# Update matplotlib parameters with these settings
plt.rcParams.update(tex_fonts)