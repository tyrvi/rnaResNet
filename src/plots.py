import pandas as pd
# from ggplot import *
import seaborn as sns
from matplotlib import pyplot as plt

# def scatter_plot(df1, df2, x='PC1', y='PC2', title=""):
#     df = pd.concat([df1, df2])
#     p = ggplot(df, aes(x=x, y=y, color='tissue', shape='study')) + geom_point() + ggtitle(title)
#     # print(p)
#     return p
    
def heatmap(df1, df2, title=""):
    df = pd.concat([df1.drop(["study", "tissue"], axis=1), 
                    df2.drop(["study", "tissue"], axis=1)])
    df = df.transpose()
    corr = df.corr()
    # plt.style.use('ggplot')
    # plt.figure(figsize=(12, 10))
    # ax = plt.axes()
    # ax = sns.heatmap(corr, xticklabels=False, yticklabels=False, ax=ax)
    ax = sns.heatmap(corr, xticklabels=False, yticklabels=False)
    ax.set_title(title)
    # plt.show()
    return ax
        
# def hist(df1, df2, x="PC1", title=""):
#     df = pd.concat([df1, df2])
#     p = ggplot(df, aes(x=x)) + geom_histogram()
#     # print(p)
#     return p
