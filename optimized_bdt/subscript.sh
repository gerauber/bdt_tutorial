#!/bin/bash

python correlations.py 
wait $!

methods=('basic_rank'
         'advanced_rank'
         'rfe')
for i in {0..2}; do
    python ranking.py -m ${methods[i]}
    wait $!
done

methods=('bayesian'
         'random'
         'optuna')
for i in {0..2}; do
    python hyperparameters.py -m ${methods[i]}
    wait $!
done
