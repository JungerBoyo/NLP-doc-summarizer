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
#    for method in range(7):
    for method in range(12):
        method_results_rouge = []
        method_results_bert = []
        for value in data:
            print(f'{value} --- {method}')
            
            method_results_rouge.append(data[value][method]['ROUGE']['rougeL'])
            method_results_bert.append(data[value][method]['BERTScore']['F1'])
        methods_results.append([method_results_rouge, method_results_bert])

    #fig, axes = plt.subplots(1, 2)
    fig, axes = plt.subplots(2, 1)
    width = 0.1
#    colors = [ 'lightgreen', 'lightblue', 'salmon', 'palegoldenrod', 'lightpink', 'lightyellow', 'lightgray' ]
#    labels = [ 'Naive', 'First last', 'TF-IDF', 'Text Rank', 'T5', 'Pegasus', 'BART', ]
    colors = [ 
        'lightgreen', 'lightblue', 'salmon', 'palegoldenrod', 'lightpink', 'lightyellow', 'lightgray', 'lavender', 'skyblue', 'peachpuff', 'khaki', 'plum' 
    ]
    labels = [ 'Naive', 'First last', 'TF-IDF', 'Text Rank', 'T5', 'Pegasus', 'BART', 'Text Rank+First Last', 'Text Rank+TF IDF+First Last', 'Text Rank+TF IDF',
              'T5+Pegasus', 'Pegasus+Bart']

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
            methods_results[3][i],
            methods_results[4][i],
            methods_results[5][i],
            methods_results[6][i], 
            methods_results[7][i],
            methods_results[8][i],
            methods_results[9][i],
            methods_results[10][i],
            methods_results[11][i]
        ]
        print(methods_results[0][i])
        print(methods_results[1][i])
        print(methods_results[2][i])
        print(methods_results[3][i])
        print(methods_results[4][i])
        print(methods_results[5][i])
        print(methods_results[6][i])
        print(methods_results[7][i])
        print(methods_results[8][i])
        print(methods_results[9][i])
        print(methods_results[10][i])
        print(methods_results[11][i])

        positions = [
            1 - 5*width - width/2,
            1 - 4*width - width/2,
            1 - 3*width - width/2,
            1 - 2*width - width/2,
            1 - width - width/2,
            1 -         width/2,
            1 +         width/2, 
            1 + width + width/2,
            1 + 2*width + width/2,
            1 + 3*width + width/2,
            1 + 4*width + width/2,
            1 + 5*width + width/2
        ]

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
