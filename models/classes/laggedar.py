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
    def __init__(self, data_path, n_days, n_test_days, n_lags=1):
        self.n_lags = n_lags
        self.time_index_flag = True
        super().__init__(name='LaggedARModel', data_path=data_path, n_days=n_days, n_test_days=n_test_days, cat_cols=[], num_cols=[])

    def set_data(self):
        super().set_data()
        for lag in range(1, self.n_lags + 1):
            self.data[f'lag_{lag}'] = self.data['revenue'].shift(lag)
            self.data[f'lag_diff_{lag}'] = self.data['revenue_diff'].shift(lag)
        self.data = self.data.dropna()

    def define_model(self):
        with pm.Model() as self.model:
            # Priors for the intercept and linear regression coefficients
            intercept = pm.Normal('intercept', mu=0, sigma=10)
            betas = pm.Normal('betas', mu=0, sigma=10, shape=self.n_lags)

            # observed data
            R = pm.MutableData("revenue", self.data.revenue.to_numpy())
            R_diff = pm.MutableData("revenue_diff", self.data.revenue_diff.to_numpy())

            # Prepare lagged revenue_diff data
            lagged_revenue_diffs = [pm.MutableData(f"lag_diff_{lag}", self.data[f"lag_diff_{lag}"].to_numpy()) for lag in range(1, self.n_lags + 1)]

            # Calculate expected revenue_diff
            expected_revenue_diff_sum = intercept
            for i, lagged_revenue_diff in enumerate(lagged_revenue_diffs):
                expected_revenue_diff_sum += betas[i] * lagged_revenue_diff
            expected_revenue_diff = pm.Deterministic('expected_revenue_diff', expected_revenue_diff_sum)

            # Likelihood (sampling distribution) of the observations
            sigma = pm.HalfNormal('sigma', sigma=10)
            target = pm.Normal('target', mu=expected_revenue_diff, sigma=sigma, observed=R_diff)

# ----------- forecasting ------------- #
    def generate_predictions(self,include_recent=True,multiday=True):
        print(f"[TASK] {self.name} generating samples from predictive distribution given data")
        if include_recent:
            # insert the last 15 days of the training data into the test data
            self.prediction_data = pd.concat([self.data[-15:],self.test_data],ignore_index=True)
        else:
            self.prediction_data = self.test_data
        self.prediction_data.reset_index(inplace=True,drop=True)

        with self.model:
            # set data to test data in order to set the mean of the likelihood distribution
            for lag in range(1,self.n_lags+1):
                pm.set_data({f'lag_diff_{lag}': self.prediction_data[f'lag_diff_{lag}'].to_numpy()})
            pm.set_data({'revenue_diff':np.zeros(len(self.prediction_data))})# self.prediction_data['revenue_diff'].to_numpy()})

            # generate samples from predictive distribution
            self.test_posterior_predictive = pm.sample_posterior_predictive(self.trace, var_names=['target'])

            # flatten target array
            self.predictions = self.test_posterior_predictive.posterior_predictive.target.values
            print('predictions:',self.predictions.shape,self.predictions)
            self.flat_predictions = self.predictions.reshape(-1, self.predictions.shape[-1])
            print('flatten:',self.flat_predictions.shape)

            revenue_predictions = np.zeros_like(self.flat_predictions)
            print(revenue_predictions.shape)
            for i in range(1, revenue_predictions.shape[1]):
                revenue_predictions[:, i] = self.prediction_data['revenue'][i - 1] + self.flat_predictions[:, i]

        print(f"[DONE] {self.name} test samples generated from predictive distribution ")
