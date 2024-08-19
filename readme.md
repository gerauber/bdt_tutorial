# BDT Tutorial

## General comments

`bdt_tutorial` is a a set of scripts to familiarise you with all aspects 
related to Boosted Decision Trees (BDTs).
BDTs are multivariate analysis techniques utilizing machine learning algorithms.
They are employed to identify patterns and use this knowledge to predict 
outcomes for comparable, but unknown datasets.

## Procedure

### Dataset

First, it is important to have a dataset on which we want to apply our BDT.
It is composed of two slightly different sets containing the same columns.
We merge together these sets to have our dataset, which is saved in a `.pkl` file

`create_pkl.py` allows the creation of this dataset through the use of a _Parser_. 
This Parser takes as parameter the number of rows _x_ that have to be created, 
and the name _filename_ of the output files (both dataset and plot of the features).
Therefore, the user has to execute:
```
python create_pkl.py -r x -f filename
```

#### Remarks

* The dataset has a _weight_ column that will be useful later, as the BDT focuses 
on ditributions shapes, and therefore this extra piece of information can be 
decisive in the classification.



### BDT
Then, the user can start playing with all aspects surrounding BDTs.
In the notebook `bdt.ipynb`, the user can have an interactive experience.

#### Remarks

* Jupyter notebooks can be opened by typing the command `jupyter notebook`
in the terminal, such that the program will instantiate a local server at _localhost:8888_, 
and the Jupyter Notebook interface will pop up in a browser window.



### Optimization





## Good practices

This code follows the style convention established in the [PEP8 document](https://peps.python.org/pep-0008/).
