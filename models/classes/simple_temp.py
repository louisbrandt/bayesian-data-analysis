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
class SimpleTempModel(BayesianModel):
    def __init__(self, data_path,n_days,n_test_days):
        cat_cols = ['dow']
        num_cols = ['tempmax']
        super().__init__(name='LinearTemperatureDowModel', data_path=data_path, cat_cols=cat_cols, num_cols=num_cols, n_days=n_days, n_test_days=n_test_days)

    def define_model(self):
        # assumes self.data contains training data in DataFrame 

        # define model
        with pm.Model() as self.model:

            # priors
            b = pm.Normal('beta', mu=1,sigma=1)
            sigma = pm.HalfNormal('sigma',sigma=1)
            a_t = pm.Normal('alpha_temp',mu=1,sigma=1)
            a_dow = pm.Normal('alpha_dow',mu=1,sigma=1,shape=len(self.data.dow.cat.categories))

            # continuous data
            T = pm.MutableData('tempmax',self.data.tempmax.to_numpy())
            # index variable
            Dow = pm.MutableData('dow',self.data.dow.cat.codes)

            # target variable
            R = pm.MutableData('revenue',self.data.revenue.to_numpy())

            # deterministic rv
            mu = pm.Deterministic('mu', b + a_t * T + a_dow[Dow])

            # likelihoood 
            target = pm.Normal('target', mu=mu, sigma=sigma, observed=R)
            # nu = pm.Exponential('nu', lam=1/15)
            # target = pm.StudentT('target', nu=nu, mu=mu, sigma=sigma, observed=R)
