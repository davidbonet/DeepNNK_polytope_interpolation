import os
import json
import matplotlib.pyplot as plt 
import seaborn as sbn
import pandas as pd
import numpy as np
from absl import flags, app
pd.options.mode.chained_assignment = None

FLAGS = flags.FLAGS
flags.DEFINE_string("neighbor_metrics_csv", "results.csv", "Path to NNK results from various activations (layers/channels) (.csv)")
flags.DEFINE_string("layers", "pre_relu", "Which layers to plot: (pre_relu, post_relu)")


def main(arg=None):
    
    # Seaborn
    sbn.set_style("whitegrid")
    palette_total = {"0": "gold", "1": "gold", "2": "gold", "3": "gold", "4": "gold", "5": "gold", "6": "gold", 
            "7": "gold", "8": "gold", "9": "gold", "10": "gold", "11": "gold", "12": "gold",
            "13": "gold", "14": "gold", "15": "gold", "All": "cornflowerblue"}
    palette_other ={"0": "firebrick", "1": "firebrick", "2": "firebrick", "3": "firebrick", "4": "firebrick", "5": "firebrick", "6": "firebrick", 
        "7": "firebrick", "8": "firebrick", "9": "firebrick", "10": "firebrick", "11": "firebrick", "12": "firebrick",
        "13": "firebrick", "14": "firebrick", "15": "firebrick", "All": "red"}
    palette_same ={"0": "seagreen", "1": "seagreen", "2": "seagreen", "3": "seagreen", "4": "seagreen", "5": "seagreen", "6": "seagreen", 
        "7": "seagreen", "8": "seagreen", "9": "seagreen", "10": "seagreen", "11": "seagreen", "12": "seagreen",
        "13": "seagreen", "14": "seagreen", "15": "seagreen", "All": "lime"}
    palette_weights = {"0": "orange", "1": "orange", "2": "orange", "3": "orange", "4": "orange", "5": "orange", "6": "orange", 
            "7": "orange", "8": "orange", "9": "orange", "10": "orange", "11": "orange", "12": "orange",
            "13": "orange", "14": "orange", "15": "orange", "All": "blueviolet"}
    capsize = .025

    df = pd.read_csv(FLAGS.neighbor_metrics_csv)
    assert len(df.Channel.unique()) > 10 # Using network_architecture_v2()
    if FLAGS.layers == 'pre_relu':
        df = df[df.Layer % 2 == 0]
    elif FLAGS.layers == 'post_relu':
        df = df[df.Layer % 2 == 1]
    else:
        raise EnvironmentError("Unknown layer case %s. Plot all layers, only the ones before applying the ReLU, or after applying it." % FLAGS.layers)
    
    unique_queries = df.Query.unique().tolist()
    dataframes = []
    for query in unique_queries:
        dataframe = df[df.Query == query]
        dataframes.append(dataframe)
    unique_queries.append(list(unique_queries))
    dataframes.append(df)

    plot_neighbors_path = "plot_neighbor_metrics/"
    if not os.path.exists(plot_neighbors_path):
        os.makedirs(plot_neighbors_path)

    for ii in range(len(unique_queries)):

        # Plot only number of neighbors 
        print(f'Plotting Num Neighbors query {str(unique_queries[ii])}')
        plt.figure(figsize=(10,4))
        ax = sbn.barplot(x='Layer', y='Num Neighbors',hue='Channel', palette=palette_total, edgecolor=".2", data=dataframes[ii], errwidth='1.0')#, errcolor='.0')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
        plt.ylabel('Number of neighbors', fontweight='bold')
        ax.set_title('Activations ' + FLAGS.layers + '. Test queries index: ' + str(unique_queries[ii]))
        if FLAGS.layers == 'post_relu':
            plt.xticks([0, 1, 2, 3, 4, 5],['0\n(32x32x16)', '1\n(16x16x16)', '2\n(16x16x16)', '3\n(8x8x16)', '4\n(8x8x16)', '5\n(4x4x16)'])
        else:
            plt.xticks([0, 1, 2, 3, 4, 5],['0\n(32x32x16)', '1\n(32x32x16)', '2\n(16x16x16)', '3\n16x16x16)', '4\n(8x8x16)', '5\n(8x8x16)'])
        ax.set_xlabel('Layer', fontsize=10)
        plt.tight_layout()
        sbn.despine(right=True, top=True)
        ax.get_figure().savefig(os.path.join(plot_neighbors_path, FLAGS.layers+'_barplot_numNeighbors_query_'+str(unique_queries[ii]).replace(', ','_').replace('[','').replace(']','')+'.png'))
        plt.close()
        
        # Plot total number of neighbors and neighbors of same class
        print(f'Plotting Same Class Num Neighbors query {str(unique_queries[ii])}')
        plt.figure(figsize=(10,4))
        ax = sbn.barplot(x='Layer', y='Num Neighbors',hue='Channel', palette=palette_other, edgecolor=".2", data=dataframes[ii], errwidth='1.0', errcolor='.2', capsize=capsize)
        ax = sbn.barplot(x='Layer', y='Num Neighbors Same Class',hue='Channel', palette=palette_same, edgecolor=".2", data=dataframes[ii], errwidth='1.0', errcolor='.8', capsize=capsize)
        plt.legend(ncol=2, bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
        ax.set_title('Activations ' + FLAGS.layers + '. Test queries index: ' + str(unique_queries[ii]))
        plt.ylabel('Number of neighbors', fontweight='bold')
        if FLAGS.layers == 'post_relu':
            plt.xticks([0, 1, 2, 3, 4, 5],['0\n(32x32x16)', '1\n(16x16x16)', '2\n(16x16x16)', '3\n(8x8x16)', '4\n(8x8x16)', '5\n(4x4x16)'])
        else:
            plt.xticks([0, 1, 2, 3, 4, 5],['0\n(32x32x16)', '1\n(32x32x16)', '2\n(16x16x16)', '3\n16x16x16)', '4\n(8x8x16)', '5\n(8x8x16)'])
        ax.set_xlabel('Layer', fontsize=10)
        sbn.despine(right=True, top=True)
        # ax.get_legend().remove()
        plt.tight_layout()
        ax.get_figure().savefig(os.path.join(plot_neighbors_path, FLAGS.layers+'_barplot_numSameClassNeighbors_query_'+str(unique_queries[ii]).replace(', ','_').replace('[','').replace(']','')+'.png'))
        plt.close()

        # Plot weight of same class neighbors
        print(f'Plotting Weight Same Class Neighbors query {str(unique_queries[ii])}')
        plt.figure(figsize=(10,4))
        ax = sbn.barplot(x='Layer', y='Weight Same Class',hue='Channel', palette=palette_weights, edgecolor=".2", data=dataframes[ii], errwidth='1.0')#, errcolor='.0')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
        ax.set_title('Activations ' + FLAGS.layers + '. Test queries index: ' + str(unique_queries[ii]))
        plt.ylabel('NNK same class neighbors weight', fontweight='bold')
        if FLAGS.layers == 'post_relu':
            plt.xticks([0, 1, 2, 3, 4, 5],['0\n(32x32x16)', '1\n(16x16x16)', '2\n(16x16x16)', '3\n(8x8x16)', '4\n(8x8x16)', '5\n(4x4x16)'])
        else:
            plt.xticks([0, 1, 2, 3, 4, 5],['0\n(32x32x16)', '1\n(32x32x16)', '2\n(16x16x16)', '3\n16x16x16)', '4\n(8x8x16)', '5\n(8x8x16)'])
        ax.set_xlabel('Layer', fontsize=10)
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        sbn.despine(right=True, top=True)
        ax.get_figure().savefig(os.path.join(plot_neighbors_path, FLAGS.layers+'_barplot_weightSameClass_query_'+str(unique_queries[ii]).replace(', ','_').replace('[','').replace(']','')+'.png'))
        plt.close()
        
        # Plot weight of same class neighbors    
        # weight_difference = dataframes[ii].loc[:, 'Weight Same Class k-NN'].to_numpy() - dataframes[ii].loc[:, 'Weight Same Class'].to_numpy()
        dataframes[ii].loc[:,"Weight Difference"] = dataframes[ii].loc[:, 'Weight Same Class k-NN'] - dataframes[ii].loc[:, 'Weight Same Class']
        print(f'Plotting Difference between k-NN same class neighbors weight vs. NNKs, query {str(unique_queries[ii])}')
        plt.figure(figsize=(10,4))
        ax = sbn.barplot(x='Layer', y='Weight Difference',hue='Channel', palette=palette_weights, edgecolor=".2", data=dataframes[ii], errwidth='1.0')#, errcolor='.0')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
        ax.set_title('Activations ' + FLAGS.layers + '. Test queries index: ' + str(unique_queries[ii]))
        plt.ylabel("Same class neighbor weight difference between k-NN and NNK\n(Positive values mean that k-NN's weight is bigger)", fontweight='bold', fontsize=8)
        if FLAGS.layers == 'post_relu':
            plt.xticks([0, 1, 2, 3, 4, 5],['0\n(32x32x16)', '1\n(16x16x16)', '2\n(16x16x16)', '3\n(8x8x16)', '4\n(8x8x16)', '5\n(4x4x16)'])
        else:
            plt.xticks([0, 1, 2, 3, 4, 5],['0\n(32x32x16)', '1\n(32x32x16)', '2\n(16x16x16)', '3\n16x16x16)', '4\n(8x8x16)', '5\n(8x8x16)'])
        ax.set_xlabel('Layer', fontsize=10)
        plt.ylim([-1.0, 1.0])
        plt.tight_layout()
        sbn.despine(right=True, top=True)
        ax.get_figure().savefig(os.path.join(plot_neighbors_path, FLAGS.layers+'_barplot_weightDifference_query_'+str(unique_queries[ii]).replace(', ','_').replace('[','').replace(']','')+'.png'))
        plt.close()
        

if __name__ == "__main__":
    app.run(main)