import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter



def dfScatterHist(df1, df2, axis1, axis2, title=''):
    plt.style.use('ggplot')
    nullfmt = NullFormatter()
    
    sm = {'GTEX': 'o', 'TCGA': '^'}
    
    df1_shape = np.array(list(map(lambda s: sm[s], df1['study'])))
    df2_shape = np.array(list(map(lambda s: sm[s], df2['study'])))

    keys = df1['tissue'].unique()
    values = np.arange(keys.shape[0])

    cm = dict(zip(keys, values))

    df1_color = np.array(list(map(lambda c: cm[c], df1['tissue'])))
    df2_color = np.array(list(map(lambda c: cm[c], df2['tissue'])))

    x1 = df1[axis1].values
    y1 = df1[axis2].values

    x2 = df2[axis1].values
    y2 = df2[axis2].values
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    plt.figure(figsize=(8, 8))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
       
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    cmap1 = plt.cm.get_cmap('Accent',max(df1_color)-min(df1_color)+1)
    cmap2 = plt.cm.get_cmap('Accent',max(df2_color)-min(df2_color)+1)
    
    bounds = range(min(df1_color),max(df1_color)+2)
    norm = colors.BoundaryNorm(bounds, cmap1.N)
    
    axScatter.scatter(x1, y1, c=df1_color, cmap=cmap1, marker=sm['GTEX'], label='GTEX')
    axScatter.scatter(x2, y2, c=df2_color, cmap=cmap2, marker=sm['TCGA'], label='TCGA')
    
    # now determine nice limits by hand:
    binwidth = 0.5
    xymax = np.max([np.max(np.fabs(x1)), np.max(np.fabs(y1))]) + 10
    lim = (int(xymax/binwidth) + 1) * binwidth
    
#     axScatter.set_xlim((-lim, lim))
#     axScatter.set_ylim((-lim, lim))
    
    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x1, bins=bins, color = 'blue', density=True, stacked = True, histtype='step')
    axHisty.hist(y1, bins=bins, orientation='horizontal', color = 'blue', density=True, 
                 stacked = True, histtype='step')
    axHistx.hist(x2, bins=bins, color = 'red', density=True, stacked = True, histtype='step')
    axHisty.hist(y2, bins=bins, orientation='horizontal', color = 'red', density=True, 
                 stacked = True, histtype='step')
    
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    
    axHistx.set_xticklabels([])
    axHistx.set_yticklabels([])
    axHisty.set_xticklabels([])
    axHisty.set_yticklabels([])
    
    axScatter.set_xlabel(axis1, fontsize=15)
    axScatter.set_ylabel(axis2, fontsize=15)
    
    #handles, labels = axScatter.get_legend_handles_labels()
    #axScatter.legend(handles, labels, bbox_to_anchor=(1.55, 1))
    axScatter.legend()

    plt.show()
    
    
    

def multiScatterHist(x1, x2, y1, y2, colors1, colors2, axis1='', axis2='', title=''):
    plt.style.use('ggplot')
    nullfmt = NullFormatter()
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    plt.figure(figsize=(12, 12))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
       
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # the scatter plot:
    axScatter.scatter(x1, x2, color=colors1, marker='o', s=20)
    axScatter.scatter(y1, y2, color=colors2, marker='s', facecolors='none', s=20) 


    # now determine nice limits by hand:
    binwidth = 0.5
    xymax = np.max([np.max(np.fabs(x1)), np.max(np.fabs(x2))]) + 10
    lim = (int(xymax/binwidth) + 1) * binwidth
    
    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))
    
    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x1, bins=bins, color = 'blue', normed=True, stacked = True, histtype='step' )
    axHisty.hist(x2, bins=bins, orientation='horizontal', color = 'blue', normed=True, 
                 stacked = True, histtype='step')
    axHistx.hist(y1, bins=bins, color = 'red', normed=True, stacked = True, histtype='step')
    axHisty.hist(y2, bins=bins, orientation='horizontal', color = 'red', normed=True, 
                 stacked = True, histtype='step')
    
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    
    axHistx.set_xticklabels([])
    axHistx.set_yticklabels([])
    axHisty.set_xticklabels([])
    axHisty.set_yticklabels([])
    axScatter.set_xlabel(axis1, fontsize=15)
    axScatter.set_ylabel(axis2, fontsize=15)

    plt.show()