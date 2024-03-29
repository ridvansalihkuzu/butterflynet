"""
This code is generated by Ridvan Salih KUZU @DLR
LAST EDITED:  11.01.2022
ABOUT SCRIPT:
It contains some custom plotting functions for visualizing the experimental results in this project.
"""

import seaborn.categorical
seaborn.categorical._Old_Violin = seaborn.categorical._ViolinPlotter

class _My_ViolinPlotter(seaborn.categorical._Old_Violin):

    def __init__(self, *args, **kwargs):
        super(_My_ViolinPlotter, self).__init__(*args, **kwargs)
        self.gray='grey'

seaborn.categorical._ViolinPlotter = _My_ViolinPlotter

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

sns.set(rc={'figure.figsize':(5,10)})
sns.set_theme(style="whitegrid")
colors_list = ['#78C850','#A8B820','#F8D030','#E0C068', '#F08030', '#C03028', '#F85888','#6890F0'] # '#98D8D8'

res = pd.read_csv("results_1.csv")
#ax = sns.boxplot(y="model", x="result",data=res[res['metric'].isin(['PA'])], dodge=False,orient="h", palette="Blues").set(
#    xlabel='Sørensen-Dice coefficient'
#)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey='row')
score_type=['PA', 'IoU','SDC']
score_label=['Pixel Accuracy', 'Intersection-over-Union','Sørensen-Dice Coefficient']
sns.set(font_scale=1.2)
for i in range(3):
    l=res['metric'].isin([score_type[i]]) & res['status'].isin(['1'])
    #axes[i] = sns.stripplot(ax=axes[i],y="model", x="result",data=res[l], palette=colors_list, edgecolor="white", linewidth=1,  marker="D",jitter=True)
    sns.swarmplot(ax=axes[i], y="model", x="result", hue='status',
                             data=res[res['metric'].isin([score_type[i]])], dodge=False, orient="h", color='red')

    sns.violinplot(ax=axes[i],y="model", x="result", hue='status',data=res[res['metric'].isin([score_type[i]])], dodge=False,orient="h", color='red', split=True,scale="count", inner="quartile").set_xlabel(score_label[i],fontsize=16)
    #legend_elements = [Line2D([0], [0], color='b', lw=4, label='Line'),Line2D([0], [0], marker='o', color='w', label='Scatter',markerfacecolor='g', markersize=15),Line2D([0], [0], marker='o', color='w', label='Scatter',markerfacecolor='g', markersize=15),]
    #axes[i].legend(legend_elements, ['Score distribution','Score after a test run', 'Average score'])
    axes[i].legend().set_visible(False)
    axes[i].set_ylabel("")

legend_elements = [Line2D([0], [0], color='grey', lw=2, label='Line'),
                   Line2D([0], [0], marker='o', color='w', label='Scatter',markerfacecolor='black', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Scatter',markerfacecolor='red', markersize=10),]
fig.legend(legend_elements, ['Kernel density estimation of scores', 'Segmentation score after each test run', 'Average of segmentation scores'],loc='upper center', ncol = 3)
plt.autoscale()
plt.savefig("benchmark_1.png")
plt.show()


res = pd.read_csv("results_2.csv")
sns.relplot(x="NUMBER OF OPERATIONS (G-FLOPS)", y="SØRENSEN-DICE COEFFICIENT", hue="BENCHMARK MODEL NAME", size="NUMBER OF PARAMETERS (M)",
            sizes=(30, 1000), alpha=.9, palette=colors_list,
            height=4, data=res[res['metric'].isin(['SDC'])])

plt.autoscale()
plt.legend([],[], frameon=False)
plt.show()