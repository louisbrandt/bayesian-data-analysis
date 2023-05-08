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

# -------------- Weather GLM ----------------- #
class WeatherGLMv2(BayesianModel):
    def __init__(self, n_days):
        # cat_cols = ['precip','n_stores','dow','day','month','year']
        # num_cols = ['tempmax','humidity','cloudcover','windspeed']
        cat_cols = ['precip','dow','month','year','n_stores']
        num_cols = ['tempmax']
        self.time_series_flag = False
        super().__init__(name='WeatherGLMv2', cat_cols=cat_cols, num_cols=num_cols,n_days=n_days)

    # define model
    def define_model(self):
        # assumes self.train_data contains training data in DataFrame 

        # define model
        with pm.Model() as self.model:

            # define priors
            b       = pm.Normal("beta", mu=0, sigma=1)
            sigma   = pm.HalfNormal("sigma",sigma=1) # pm.Gamma(shape=0.3, scale= 3.33)
            a_t     = pm.HalfNormal("alpha_temp",sigma=1)
            # a_h = pm.Normal("alpha_humid", mu=0, sigma=1)
            # a_c = pm.Normal("alpha_cloud", mu=0, sigma=1)
            # a_w = pm.Normal("alpha_wind", mu=0, sigma=1)

            a_p = pm.Normal("alpha_precip", mu=[1,0,-1], sigma=1,shape=len(self.train_data.precip.cat.categories))
            dow_means = np.array([-1, 0, 1, 1, 1, 0, -1]) 
            a_dow = pm.Normal('alpha_dow',mu=dow_means,sigma=1,shape=len(self.train_data.dow.cat.categories))
            # a_d = pm.Normal("alpha_day", mu=0, sigma=1,shape=len(self.train_data.day.cat.categories))
            a_m = pm.Normal("alpha_month", mu=0, sigma=1,shape=len(self.train_data.month.cat.categories))
            a_y = pm.Normal("alpha_year", mu=0, sigma=1,shape=len(self.train_data.year.cat.categories))
            a_n = pm.Normal("alpha_n", mu=0, sigma=1,shape=len(self.train_data.n_stores.cat.categories))

            # data - mutable = True
            T = pm.MutableData("tempmax", self.train_data.tempmax.to_numpy())
            # H = pm.MutableData("humidity", self.train_data.humidity.to_numpy())
            # W = pm.MutableData("windspeed", self.train_data.windspeed.to_numpy())
            # C = pm.MutableData("cloudcover", self.train_data.cloudcover.to_numpy())

            # index variables - mutable = True
            P = pm.MutableData("precip", self.train_data.precip.cat.codes)
            Dow = pm.MutableData("dow", self.train_data.dow.cat.codes)
            # D = pm.MutableData("day", self.train_data.day.cat.codes)
            M = pm.MutableData("month", self.train_data.month.cat.codes)
            Y = pm.MutableData("year", self.train_data.year.cat.codes)
            N = pm.MutableData("n_stores", self.train_data.n_stores.cat.codes)

            # target variable - mutable = True
            R = pm.MutableData("revenue", self.train_data.revenue.to_numpy())
            # R_diff = pm.MutableData("revenue_diff", self.train_data.revenue_diff.to_numpy())

            # combine the regression coefficients and variables into the linear regression equation for the mean of the likelihood function
            # mu = pm.Deterministic('mu', b + a_t * T + a_h * H + a_c * C + a_w * W + a_p[P] + a_dow[Dow] + a_d[D] + a_m[M] + a_y[Y] + a_n[N])
            mu = pm.Deterministic('mu', b + a_t * T + a_p[P] + a_dow[Dow] + a_m[M] + a_y[Y] + a_n[N])

            # define likelihood
            target = pm.Normal('target',mu=mu,sigma=sigma,observed=R)

    def generate_predictions(self,data,multiday=None):
        print(f"[TASK] {self.name} generating samples from predictive distribution given data")

        prediction_data = data.copy()

        with self.model:
            # set data to test data in order to set the mean of the likelihood distribution
            for var in self.cat_cols:
                pm.set_data({var: prediction_data[var].cat.codes})
            for var in self.num_cols:
                pm.set_data({var: prediction_data[var].to_numpy()})

            pm.set_data({'revenue': prediction_data['revenue'].to_numpy()}) # for shape

            # generate samples from predictive distribution
            test_posterior_predictive = pm.sample_posterior_predictive(self.trace, var_names=['target'])
            predictions = test_posterior_predictive.posterior_predictive.target.values

            flat_predictions = predictions.reshape(-1, predictions.shape[-1])

        revenue_predictions = np.zeros_like(flat_predictions)

        v = self.valid_data.revenue_diff.to_numpy()[-1]
        revenue_predictions[:, 0] = v + flat_predictions[:, 0]

        for i in range(1, revenue_predictions.shape[1]):
            # print the prediction of revenue to screen 
            revenue_predictions[:, i] = prediction_data['revenue'][i - 1] + flat_predictions[:, i]

        print(f"[DONE] {self.name} test samples generated from predictive distribution ")
        return revenue_predictions
