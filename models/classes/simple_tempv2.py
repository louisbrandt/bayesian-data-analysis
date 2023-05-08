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

# -------------- Simple Temp Model ----------------- #
class SimpleTempModelv2(BayesianModel):
    def __init__(self, n_days):
        cat_cols = ['dow']
        num_cols = ['tempmax']
        self.time_series_flag = False
        super().__init__(name='SimpleTempv2', cat_cols=cat_cols, num_cols=num_cols, n_days=n_days)

    def define_model(self):
        # assumes self.train_data contains training data in DataFrame 

        # define model
        with pm.Model() as self.model:

            # informative priors
            b = pm.Normal('beta', mu=0,sigma=1)
            sigma = pm.HalfNormal('sigma',sigma=1)
            a_t = pm.HalfNormal('alpha_temp',sigma=1)
            dow_means = np.array([-1, 0, 1, 1, 1, 0, -1]) 
            a_dow = pm.Normal('alpha_dow',mu=dow_means,sigma=1,shape=len(self.train_data.dow.cat.categories))

            # continuous data
            T = pm.MutableData('tempmax',self.train_data.tempmax.to_numpy())
            # index variable
            Dow = pm.MutableData('dow',self.train_data.dow.cat.codes)

            # target variable
            R = pm.MutableData('revenue',self.train_data.revenue.to_numpy())

            # deterministic rv
            mu = pm.Deterministic('mu', b + a_t * T + a_dow[Dow])

            # likelihoood 
            target = pm.Normal('target', mu=mu, sigma=sigma, observed=R)
