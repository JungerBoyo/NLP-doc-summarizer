import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json_results_path', type=str, help="""
            Json path from which to present the results.
            """)
    args = parser.parse_args()
    
    with open(args.json_results_path, 'r') as file:
        data = json.load(file)

    methods_results = []
    for method in range(4):
        method_results_rouge = []
        method_results_bert = []
        for value in data:
            method_results_rouge.append(data[value][method]['ROUGE']['rougeL'])
            method_results_bert.append(data[value][method]['BERTScore']['F1'])
        methods_results.append([method_results_rouge, method_results_bert])

    fig, axes = plt.subplots(1, 2)
    width = 0.1
    colors = [ 'lightgreen', 'lightblue', 'salmon', 'palegoldenrod' ]
    labels = [ 'Naive', 'First last', 'TF-IDF', 'Text Rank' ]

    handles = [Patch(facecolor=color, edgecolor='black', label=label) for color, label in zip(colors, labels)]
    handles.append(Patch(facecolor='purple', edgecolor='black', label='Mean'))
    handles.append(Patch(facecolor='black', edgecolor='black', label='Median'))
    fig.legend(handles=handles, title="Legend", loc='upper right')
    
    xs = [ 'rougeL (ROUGE)', 'F1 (BERTScore)' ]
    i = 0
    for x, axis in zip(xs, axes):
        boxes = [
            methods_results[0][i],
            methods_results[1][i],
            methods_results[2][i],
            methods_results[3][i]
        ]
        print(methods_results[0][i])
        print(methods_results[1][i])
        print(methods_results[2][i])
        print(methods_results[3][i])

        positions = [1 - width - width/2, 1 - width/2, 1 + width/2, 1 + width + width/2]

        bxp = axis.boxplot(
                boxes,
                positions=positions, widths=0.1, patch_artist=True, showfliers=True,
                showmeans=True,
                meanline=True,
                meanprops={'color': 'purple', 'linewidth': 2,
                  'marker': 'o', 'markerfacecolor': 'purple', 'markeredgecolor': 'purple', 'markersize': 2}, 
                medianprops={'color': 'black', 'linewidth': 2,
                  'marker': 's', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 2})

        for patch, color in zip(bxp['boxes'], colors):
            patch.set_facecolor(color)

        axis.set_xticks([1])
        axis.set_xticklabels([x])
        #axis.set_title('Extraction based methods comparison')
        axis.set_xlabel('Method')
        axis.set_ylabel('Metric')

        i += 1

    
    plt.legend()
    plt.show()
