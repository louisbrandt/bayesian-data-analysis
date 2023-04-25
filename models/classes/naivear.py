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

# -------------- AR Model ----------------- #
class ARModel(BayesianModel):
    def __init__(self, data_path,n_days,n_test_days):
        super().__init__(name='ARModel', data_path=data_path,n_days=n_days,n_test_days=n_test_days,cat_cols=[],num_cols=[])
        self.time_index_flag = True


    def set_data(self):
        super().set_data()
        lagged_dataframes = []
        for lag in range(1, self.n_lags + 1):
            lagged_revenue = self.data['revenue'].shift(lag)
            lagged_revenue.name = f'lag_{lag}'
            lagged_dataframes.append(lagged_revenue)
            
            lagged_revenue_diff = self.data['revenue_diff'].shift(lag)
            lagged_revenue_diff.name = f'lag_diff_{lag}'
            lagged_dataframes.append(lagged_revenue_diff)
        self.data = pd.concat([self.data] + lagged_dataframes, axis=1)
        self.data = self.data.dropna()


    def define_model(self):

        with pm.Model() as self.model:
            # Priors for the intercept and linear regression coefficients
            intercept = pm.Normal('intercept', mu=0, sigma=10)
            betas = pm.Normal('betas', mu=0, sigma=10, shape=self.n_lags)

            # observed data
            R = pm.MutableData("revenue", self.data.revenue.to_numpy())

            # Prepare lagged revenuedata
            lagged_revenue = [pm.MutableData(f"lag_{lag}", self.data[f"lag_{lag}"].to_numpy()) for lag in range(1, self.n_lags + 1)]

            # Calculate expected revenue
            expected_revenue_sum = intercept
            for i, lagged_revenue in enumerate(lagged_revenues):
                expected_revenue_sum += betas[i] * lagged_revenue
            expected_revenue= pm.Deterministic('expected_revenue', expected_revenue_sum)

            # Likelihood (sampling distribution) of the observations
            sigma = pm.HalfNormal('sigma', sigma=10)
            target = pm.Normal('target', mu=expected_revenue, sigma=sigma, observed=R)

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
                pm.set_data({f'lag_{lag}': self.prediction_data[f'lag_{lag}'].to_numpy()})
            pm.set_data({'revenue':np.zeros(len(self.prediction_data))})# self.prediction_data['revenue_diff'].to_numpy()})

            # generate samples from predictive distribution
            self.test_posterior_predictive = pm.sample_posterior_predictive(self.trace, var_names=['target'])

            # flatten target array
            self.predictions = self.test_posterior_predictive.posterior_predictive.target.values
            self.flat_predictions = self.predictions.reshape(-1, self.predictions.shape[-1])

        print(f"[DONE] {self.name} test samples generated from predictive distribution")
