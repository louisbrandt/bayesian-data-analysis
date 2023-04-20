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

    def define_model(self):

        with pm.Model() as self.model:
            # Priors for the linear regression coefficients
            intercept = pm.Normal('intercept', mu=0, sigma=10)
            beta = pm.Normal('beta', mu=0, sigma=10)

            # target variable - mutable = True
            R = pm.MutableData("revenue", self.data.revenue.to_numpy())

            # Set the test data time index
            time_index = pm.MutableData("time_index", np.arange(self.n_days))

            # Expected value of the revenue based on the linear regression
            expected_revenue = pm.Deterministic('expected_revenue',intercept + beta * time_index)

            # Likelihood (sampling distribution) of the observations
            sigma = pm.HalfNormal('sigma', sigma=10)
            target = pm.Normal('target', mu=expected_revenue, sigma=sigma, observed=R)
