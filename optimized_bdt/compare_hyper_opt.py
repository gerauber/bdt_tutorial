#############################
## author: G. Räuber, 2024 ##
#############################

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filenames', nargs='+',
                    default=['hyperparams_random',
                             'hyperparams_bayesian',
                             'hyperparams_optuna'],
                    type=list, help='Name of input files')
parser.add_argument('-o', '--outname', default='compare_hyperparams', type=str,
                    help='Name of onput files')
args = parser.parse_args()


# Collect the inputs
def dictfilt(x, y):
    return {i: x[i] for i in x if i in set(y)}


dict_input = {}
for na in args.filenames:
    with open(f'{na}.json') as json_file:
        dict_input[na] = dictfilt(json.load(json_file),
                                  ('best_score', 'best_estimate'))
        dict_keys = list(dict_input[na]['best_estimate'].keys())
        for ke in dict_keys:
            dict_input[na][ke] = dict_input[na]['best_estimate'][ke]
        dict_input[na].pop('best_estimate', None)

# Extract all unique parameter names
parameters = list(dict_input['hyperparams_random'].keys())

# Number of parameters
num_params = len(parameters)

# Define the layout: three columns, calculate the number of rows needed
ncols = 3
nrows = (num_params + 1) // ncols

# Create subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop over each parameter and create a subplot
for i, param in enumerate(parameters):
    random_val = dict_input[args.filenames[0]].get(param, 0)
    bayesian_val = dict_input[args.filenames[1]].get(param, 0)
    optuna_val = dict_input[args.filenames[2]].get(param, 0)

    # Bar positions for this subplot
    bar_positions = np.arange(3)
    bar_values = [random_val, bayesian_val, optuna_val]

    # Plotting the bars horizontally
    bars = axes[i].barh(bar_positions, bar_values,
                        color=['#2145b882', '#e67500a3', '#8abd66c2'])

    # Set y-ticks and labels
    axes[i].set_yticks(bar_positions)
    axes[i].set_yticklabels(['Random', 'Bayesian', 'Optuna'])

    # Set title and labels
    axes[i].set_xlabel('Value')
    axes[i].set_title(f'{param}', fontsize=15)

    # Set grid
    axes[i].grid(True, color='whitesmoke')
    axes[i].set_axisbelow(True)

    # Annotate the bars with their values
    for bar, value in zip(bars, bar_values):
        width = bar.get_width()
        # If the value is large compared to others
        if width > max(bar_values) * 0.8:
            axes[i].text(width - 0.02 * max(bar_values),
                         bar.get_y() + bar.get_height() / 2,
                         f'{value:.1e}', va='center', ha='right',
                         fontsize=12, color='white')
        # If the value is small compared to others
        else:
            axes[i].text(width + 0.02 * max(bar_values),
                         bar.get_y() + bar.get_height() / 2,
                         f'{value:.1e}', va='center', ha='left',
                         fontsize=12, color='black')

# Turn off any empty subplots if the number of parameters is not
# a multiple of ncols
for j in range(i+1, len(axes)):
    axes[j].axis('off')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save plot
plt.savefig(f"{args.outname}.pdf", bbox_inches='tight')
