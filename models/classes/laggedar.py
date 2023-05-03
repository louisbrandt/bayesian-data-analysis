import os
import pickle
import cloudpickle
import scipy.stats as stats
import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import aesara.tensor as at
import xarray as xr
from bayesian import BayesianModel

class LaggedARModel(BayesianModel):
    def __init__(self, n_days, n_lags):
        self.n_lags = n_lags
        self.time_series_flag = True
        super().__init__(name='LaggedARModel', n_days=n_days, cat_cols=[], num_cols=[])

    def add_lags(self, train_data, valid_data, test_data):
        d = pd.concat([train_data, valid_data, test_data],ignore_index=True)
        lagged_dataframes = []
        for lag in range(1, self.n_lags + 1):
            lagged_revenue = d['revenue'].shift(lag)
            lagged_revenue.name = f'lag_{lag}'
            lagged_dataframes.append(lagged_revenue)
            
            lagged_revenue_diff = d['revenue_diff'].shift(lag)
            lagged_revenue_diff.name = f'lag_diff_{lag}'
            lagged_dataframes.append(lagged_revenue_diff)
        data = pd.concat([d] + lagged_dataframes, axis=1)
        data = data.dropna() 
        test = data[-21:]
        valid = data[-42:-21]
        train = data[:-42]
        return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)

    def set_data(self):
        train_data = pd.read_csv('/Users/louisbrandt/itu/6/bachelor/data/processed/train.csv')
        valid_data = pd.read_csv('/Users/louisbrandt/itu/6/bachelor/data/processed/valid.csv')
        test_data = pd.read_csv('/Users/louisbrandt/itu/6/bachelor/data/processed/test.csv')

        self.train_data, self.valid_data, self.test_data = self.add_lags(train_data, valid_data, test_data)

        self.train_data = self.train_data.astype({col: 'category' for col in self.cat_cols})
        self.valid_data = self.valid_data.astype({col: 'category' for col in self.cat_cols})
        self.test_data = self.test_data.astype({col: 'category' for col in self.cat_cols})

        self.train_data = self.train_data[-self.n_days:]

    def define_model(self):
        with pm.Model() as self.model:
            # Priors for the intercept and linear regression coefficients
            intercept = pm.Normal('intercept', mu=0, sigma=1)
            betas = pm.Normal('betas', mu=0, sigma=1, shape=self.n_lags)

            # observed data
            R = pm.MutableData("revenue", self.train_data.revenue.to_numpy())
            R_diff = pm.MutableData("revenue_diff", self.train_data.revenue_diff.to_numpy())

            # Prepare lagged revenue_diff data
            lagged_revenue_diffs = [pm.MutableData(f"lag_diff_{lag}", self.train_data[f"lag_diff_{lag}"].to_numpy()) for lag in range(1, self.n_lags + 1)]

            # Calculate expected revenue_diff
            expected_revenue_diff_sum = intercept
            for i, lagged_revenue_diff in enumerate(lagged_revenue_diffs):
                expected_revenue_diff_sum += betas[i] * lagged_revenue_diff
            expected_revenue_diff = pm.Deterministic('expected_revenue_diff', expected_revenue_diff_sum)

            # Likelihood (sampling distribution) of the observations
            sigma = pm.HalfNormal('sigma', sigma=1)
            target = pm.Normal('target', mu=expected_revenue_diff, sigma=sigma, observed=R_diff)

    def generate_predictions(self,data,multiday=None):
        print(f"[TASK] {self.name} generating samples from predictive distribution given data")

        prediction_data = data.copy()

        with self.model:
            # set data to test data in order to set the mean of the likelihood distribution
            for lag in range(1,self.n_lags+1):
                pm.set_data({f'lag_diff_{lag}': prediction_data[f'lag_diff_{lag}'].to_numpy()})
            pm.set_data({'revenue_diff':np.zeros(len(prediction_data))}) #prediction_data['revenue_diff'].to_numpy()})

            # generate samples from predictive distribution
            test_posterior_predictive = pm.sample_posterior_predictive(self.trace, var_names=['target'])
            predictions = test_posterior_predictive.posterior_predictive.target.values
            flat_predictions = predictions.reshape(-1, predictions.shape[-1])

            revenue_predictions = np.zeros_like(flat_predictions)
            revenue_predictions[:, 0] = prediction_data.lag_1[0] + flat_predictions[:, 0]
            for i in range(1, revenue_predictions.shape[1]):
                # print the prediction of revenue to screen 
                revenue_predictions[:, i] = prediction_data['revenue'][i - 1] + flat_predictions[:, i]

        print(f"[DONE] {self.name} test samples generated from predictive distribution ")

        # print(revenue_predictions)
        return revenue_predictions
