#!/bin/bash

models=("simpletemp" "weatherglm")
n_days=("90" "180" "365" "1166")
# n_lags=("1" "7" "30" "90")

for model in "${models[@]}"; do
  for n_day in "${n_days[@]}"; do
    # for n_lag in "${n_lags[@]}"; do
    python run_model.py "$model" --n_days "$n_day" --test 
    # done
  done
done

