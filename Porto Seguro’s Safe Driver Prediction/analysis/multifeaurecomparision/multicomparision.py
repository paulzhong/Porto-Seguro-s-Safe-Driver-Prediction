"""
After studying each feature individually we will now
start to look at interactions between them. The annoymity
of the features will make it more difficult to interpret
these relations. However, they will still be useful for
our prediction goal and for gaining a more detailed
understanding of our data.
"""

import matplotlib.pyplot as plt
import seaborn as sns


class MuiltiFeatureComparisons(object):

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def HeatMap(self,x=True):
        correlations = self.dataframe.corr()
        ## Create color map ranging between two colors
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',square=True, linewidths=.5, annot=x, cbar_kws={"shrink": .75})
        fig.set_xticklabels(fig.get_xticklabels(), rotation = 90, fontsize = 10)
        fig.set_yticklabels(fig.get_yticklabels(), rotation = 0, fontsize = 10)
        plt.tight_layout()
        plt.show()

    
