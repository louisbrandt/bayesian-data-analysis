#!/bin/bash

models=("simpletempmodel" "weatherglm" "laggedarmodel")
n_days=("90" "180" "365" "1190")
n_test_days=("21")

for model in ${models[@]}; do
    for n_day in ${n_days[@]}; do
      python run_model.py "$model" --n_days "$n_day" --sample-posterior --test --include-recent 
    done
done
