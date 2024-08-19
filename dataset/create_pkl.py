import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rows', default=100000, type=int,
                    help='Number of entries')
parser.add_argument('-f', '--filename', default='dataset_2classes',
                    type=str, help='name of output files')
args = parser.parse_args()

# Have a first set of variables, with specific distributions
# Set the number of rows
num_rows = int(args.rows/2)

# Create the columns
data_1 = {
    'a': np.random.normal(loc=6, scale=1, size=num_rows),
    'b': np.random.normal(loc=1, scale=1, size=num_rows),
    'c': np.random.uniform(0, 30, size=num_rows),
    'd': np.random.poisson(lam=500.0, size=num_rows),
    'e': np.random.exponential(scale=3.0, size=num_rows),
    'f': np.random.exponential(scale=1.0, size=num_rows),
    'g': np.random.binomial(500, 0.5, size=num_rows),
    'h': np.random.rayleigh(scale=1.0, size=num_rows),
    'i': np.random.laplace(loc=0.0, scale=1.0, size=num_rows),
    'j': np.random.power(3, size=num_rows),
}

# Create the DataFrame
df_1 = pd.DataFrame(data_1)

# Add other columns (set dataset type and weight)
df_1['dataset'] = 1
df_1['weight'] = 1

# Have a second set of variables, with slightly different distributions
# Set the number of rows
if args.rows % 2 == 1:
    num_rows += 1

# Create the columns
data_2 = {
    'a': np.random.normal(loc=6.5, scale=1, size=num_rows),
    'b': np.random.normal(loc=0.8, scale=1, size=num_rows),
    'c': np.random.uniform(1, 30.5, size=num_rows),
    'd': np.random.poisson(lam=489.0, size=num_rows),
    'e': np.random.exponential(scale=3.2, size=num_rows),
    'f': np.random.exponential(scale=0.8, size=num_rows),
    'g': np.random.binomial(505, 0.49, size=num_rows),
    'h': np.random.rayleigh(scale=1.4, size=num_rows),
    'i': np.random.laplace(loc=0.5, scale=1.3, size=num_rows),
    'j': np.random.power(3.9, size=num_rows),
}

# Create the DataFrame
df_2 = pd.DataFrame(data_2)

# Have a quick look at the distributions
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 8))
axes = axes.flatten()

nbbins = 15
for i, column in enumerate(df_2.columns):
    low = min(min(df_1[column]), min(df_2[column]))
    high = max(max(df_1[column]), max(df_2[column]))
    axes[i].hist(df_1[column], bins=nbbins, alpha=0.5, range=(low, high),
                 label='df1', color='#2145b882')
    axes[i].hist(df_2[column], bins=nbbins, alpha=0.5, range=(low, high),
                 label='df2', color='#e67500a3')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel(f'Counts / {(high-low)/nbbins:<0.2f}')
    axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axes[i].legend()

for i in range(10, 12):
    axes[i].axis('off')

plt.tight_layout()
fig = plt.savefig(f'{args.filename}_features.pdf', bbox_inches='tight')
plt.show()

# Add other columns (set dataset type and weight)
df_2['dataset'] = 2
df_2['weight'] = 1

# Combine both datasets into one
# Stack the dataframes
df = pd.concat([df_1, df_2], ignore_index=True, axis=0)

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

# Display the DataFrame
print(df)

# Save the Dataframe
pickle.dump(df, open(f'{args.filename}.pkl', "wb"))
