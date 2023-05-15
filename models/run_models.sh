#!/bin/bash
models=("hybridmodel" "laggedar")
n_days=("90")
n_lags=("90")

for model in "${models[@]}"; do
  for n_day in "${n_days[@]}"; do
    for n_lag in "${n_lags[@]}"; do
      python run_model.py "$model" --n_days "$n_day" --n_lags "$n_lag" --test 
    done
  done
done

