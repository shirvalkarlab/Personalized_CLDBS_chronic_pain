# -*- coding: utf-8 -*-
"""
@author: Jackson Cagle, University of Florida
"""

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy import stats

from matplotlib import dates, colors

def getHexColor(cmap):
    return colors.rgb2hex(cmap)

def stderr(data, axis=0):
    return np.std(data, axis=axis)/np.sqrt(data.shape[axis])

def shadedErrorBar(x, y, shadedY, lineprops=dict(), alpha=1.0, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    
    line = ax.plot(x,y)[0]
    shade = ax.fill_between(x, y-shadedY, y+shadedY, alpha=alpha)
    
    for key in lineprops.keys():
        if key == "color":
            line.set_color(lineprops[key])
            shade.set_color(lineprops[key])
        elif key == "alpha":
            shade.set_alpha(lineprops[key])
        elif key == "linewidth":
            line.set_linewidth(lineprops[key])
        elif key == "linestyle":
            line.set_linestyle(lineprops[key])
    
    return (line, shade)

def surfacePlot(X, Y, Z, cmap=plt.get_cmap("jet"), ax=None):
    if ax is None:
        ax = plt.gca()
        
    bound = (X[0], X[-1], Y[0], Y[-1])
    image = ax.imshow(Z, cmap=cmap, aspect="auto", origin="lower", extent=bound, interpolation="gaussian")
    ax.set_ylim(Y[0], Y[-1])
    
    return image

def addColorbar(ax, image, title, padding=5.5):
    colorAxes = inset_axes(ax, width="2%", height="100%", loc="right", bbox_to_anchor=(0.05, 0, 1, 1), bbox_transform=ax.transAxes)
    colorbar = plt.colorbar(image, cax=colorAxes)
    #colorAxes.set_title(title,rotation=-90, loc="right", verticalalignment="center")
    colorAxes.set_ylabel(title, rotation=-90, verticalalignment="bottom")
    return colorAxes
    
def addExternalLegend(ax, lines, legends, fontsize=12):
    legendAxes = inset_axes(ax, width="2%", height="100%", loc="right", bbox_to_anchor=(0.15, 0.05, 1, 1), bbox_transform=ax.transAxes)
    legendAxes.legend(lines, legends, frameon=False, fontsize=fontsize)
    legendAxes.axis("off")
    return legendAxes
    
def singleViolin(x, y, width=0.5, showmeans=False, showextrema=True, showmedians=False, vert=True, color=None, ax=None):
    if ax is None:
        ax = plt.gca()
        
    violin = ax.violinplot(y, positions=[x], widths=[width], showmeans=showmeans, showextrema=showextrema, showmedians=showmedians, vert=vert)
    
    if color != None:
        lineParts = list()
        if showextrema:
            lineParts.append("cbars")
            lineParts.append("cmins")
            lineParts.append("cmaxes")
        if showmedians:
            lineParts.append("cmedians")
        if showmeans:
            lineParts.append("cmeans")
        
        for pc in lineParts:
            vp = violin[pc]
            vp.set_edgecolor(color)
            vp.set_linewidth(2)
        
        for pc in violin["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.4)
            
    return violin

def standardErrorBox(x, y, width=0.4, color=None, sym="o", flier_size=2, flier_edgewidth=0.25, showfliers=True, flierprops=None, ax=None):
    """
    Draw a boxplot at position x for data y with customizable fliers and black median/whiskers.
    
    Parameters:
        x                : x position of the box
        y                : data array
        width            : width of the box
        color            : fill color of the box
        sym              : symbol for outliers
        flier_size       : size of outlier symbols
        flier_edgewidth  : line width of outlier symbol edges
        showfliers       : whether to show outliers
        flierprops       : custom matplotlib flierprops dict
        ax               : matplotlib axis
    """
    if ax is None:
        ax = plt.gca()
    
    # Define flier properties if none provided
    if flierprops is None:
        flierprops = dict(marker=sym, color='black', markersize=flier_size, markeredgewidth=flier_edgewidth)
    
    # Create boxplot
    box = ax.boxplot(
        y,
        positions=[x],
        widths=[width],
        showfliers=showfliers,
        sym=sym,
        patch_artist=True,
        flierprops=flierprops
    )
    
    # Set box face color if specified
    if color is not None:
        for patch in box['boxes']:
            patch.set(facecolor=color)
    
    # Set median line color to black
    for median in box['medians']:
        median.set(color='black', linewidth=1)
    
    # Set whiskers to black
    for whisker in box['whiskers']:
        whisker.set(color='black', linewidth=0.75)
    
    # Set caps to black
    for cap in box['caps']:
        cap.set(color='black', linewidth=0.75)
        
    return box
    

def addStatisticBar(data, ax=None):
    if ax is None:
        ax = plt.gca()
    
    totalGroups = len(data)
    totalTests = sum(range(totalGroups))

    SignificantStatistc = list()
    for index1 in range(totalGroups):
        for index2 in range(index1 + 1, totalGroups):
            pvalue = stats.ttest_ind(data[index1]["Y"],data[index2]["Y"],equal_var=False).pvalue
            if pvalue * totalTests < 0.01:
                SignificantStatistc.append({"pvalue": pvalue*totalTests, "Group1": [data[index1]["X"],max(data[index1]["Y"])], "Group2": [data[index2]["X"],max(data[index2]["Y"])]})

    existingBarHeight = np.zeros(len(SignificantStatistc))
    lineIndex = 0
    for stat in SignificantStatistc:
        proposedHeight = max([stat["Group1"][1],stat["Group2"][1]]) + 1
        while np.any(existingBarHeight == proposedHeight):
            proposedHeight += 1
        existingBarHeight[lineIndex] = proposedHeight
        lineIndex += 1
        ax.plot([stat["Group1"][0],stat["Group2"][0]],[proposedHeight,proposedHeight], "k", linewidth=2)
        
        sigStars = "*"
        if stat["pvalue"] < 0.001:
            sigStars += "*"
        if stat["pvalue"] < 0.0001:
            sigStars += "*"
        ax.text(np.mean([stat["Group1"][0],stat["Group2"][0]]), proposedHeight+0.1, sigStars, fontsize=12, horizontalalignment="center", verticalalignment="center")

    return SignificantStatistc

def largeFigure(n, resolution=(1600,900), dpi=100.0):
    if n == 0:
        figure = plt.figure(figsize=(resolution[0]/dpi,resolution[1]/dpi), dpi=dpi)
    else:
        figure = plt.figure(n, figsize=(resolution[0]/dpi,resolution[1]/dpi), dpi=dpi)
    
    figure.clf()
    return figure

def imagesc(x, y, z, clim=None, cmap=plt.get_cmap("jet"), interpolation="gaussian", ax=None):
    bound = (x[0], x[-1], y[0], y[-1])

    if ax:
        image = ax.imshow(z, cmap=cmap, aspect="auto", origin="lower", extent=bound, interpolation=interpolation, interpolation_stage="rgba")
    else:
        image = plt.imshow(z, cmap=cmap, aspect="auto", origin="lower", extent=bound, interpolation=interpolation, interpolation_stage="rgba")
    if clim:
        image.set_clim(clim[0],clim[1])
    return image

def addAxes(fig):
    return fig.add_axes([0.1,0.1,0.8,0.8])

def colorTextFromCmap(color):
    if type(color) == str:
        colorInfoString = color.split(",")
        colorInfoString = [string.replace("rgb(","").replace(")","") for string in colorInfoString]
        colorInfo = [int(i) for i in colorInfoString]
    else:
        colorInfo = np.array(color[:-1]) * 255
    colorText = f"#{hex(int(colorInfo[0])).replace('0x',''):0>2}{hex(int(colorInfo[1])).replace('0x',''):0>2}{hex(int(colorInfo[2])).replace('0x',''):0>2}"
    return colorText
