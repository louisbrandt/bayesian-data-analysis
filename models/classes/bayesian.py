import json
import os
import pickle
import cloudpickle
import scipy.stats as stats
import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
import aesara.tensor as at
import xarray as xr
from scipy.stats import gaussian_kde

# ----------- Parent Class ------------- #
class BayesianModel():
    def __init__(self,name,cat_cols,num_cols,n_days):
        self.name = name
        self.cat_cols = cat_cols if cat_cols is not None else []
        self.num_cols = num_cols if num_cols is not None else []
        self.n_days = n_days
        self.set_data()
        self.define_model()
        # add hyperparams to the pickle & plot path at the end of the string
        self.pkl_path = '/Users/louisbrandt/itu/6/bachelor/pickles/'+self.name+'/'
        self.eval_path = '/Users/louisbrandt/itu/6/bachelor/eval/'+self.name+'/'
        self.pkl_path += f"{n_days}"
        self.eval_path += f"{n_days}"
        if self.time_series_flag:
            self.pkl_path += f'_{self.n_lags}'
            self.eval_path += f'_{self.n_lags}'
        self.pkl_path += '/'
        self.eval_path += '/'
        # Check if directories exist and create them if not
        for path in [self.pkl_path, self.eval_path, self.eval_path+'valid/', self.eval_path+'test/']:
            if not os.path.exists(path):
                os.makedirs(path)

    def set_data(self):
        self.train_data = pd.read_csv('/Users/louisbrandt/itu/6/bachelor/data/processed/train.csv')
        self.valid_data = pd.read_csv('/Users/louisbrandt/itu/6/bachelor/data/processed/valid.csv')
        self.test_data = pd.read_csv('/Users/louisbrandt/itu/6/bachelor/data/processed/test.csv')

        self.train_data = self.train_data.astype({col: 'category' for col in self.cat_cols})
        self.valid_data = self.valid_data.astype({col: 'category' for col in self.cat_cols})
        self.test_data = self.test_data.astype({col: 'category' for col in self.cat_cols})

        self.train_data = self.train_data[-self.n_days:]
    
    def define_model(self):
        raise NotImplementedError()
    
# ----------- sample trace ------------- #
    def sample_posterior(self, draws=1000, tune=1000, chains=4, target_accept=0.9):
        print(f"\n[TASK] ----------- Model: {self.name} \n\tsampling posterior...")
        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept,idata_kwargs={"log_likelihood": True})
        print(f"[DONE] {self.name} posterior sampling")

# ----------- store trace ------------- #
    def save_trace(self):
        with open(self.pkl_path+'trace.pkl', 'wb') as f:
            cloudpickle.dump({'model': self.model, 'trace': self.trace,'y_obs':self.train_data}, f)
        print(f"[DONE] {self.name} trace saved to {self.pkl_path}")
    
    def load_trace(self):
        print(f"\n[TASK] ----------- Model: {self.name} \n\tloading trace...")
        print(self.pkl_path+'trace.pkl')
        with open(self.pkl_path+'trace.pkl', 'rb') as f:
            saved_model = cloudpickle.load(f)
        self.trace = saved_model['trace'] 
        print(f"[DONE] {self.name} trace loaded from {self.pkl_path}")

# ----------- plot trace ------------- #
    def summary(self,save=True,show=False):
        print('[TASK] computing summary')
        summary = az.summary(self.trace)
        dont_want = summary.index[summary.index.str.startswith('mu')]
        summary = summary.loc[~summary.index.isin(dont_want)]
        if save:
            summary.to_csv(self.eval_path+'summary.csv')
            print(f"[DONE] {self.name} saved summary to file {self.eval_path}summary.csv")
        if show:
            print(summary,flush=True)

    def plot_trace(self,var_names=None,save=True,show=False):
        print(f"[TASK] {self.name} plotting trace")
        fig, ax = plt.subplots()
        with self.model:
            az.plot_trace(self.trace, var_names=var_names)
        if save:
            plt.savefig(self.eval_path+'trace.png')  # save the plot to a file
            print(f"[DONE] {self.name} saved trace to file {self.eval_path}trace.png")
        if show:
            plt.show()  # display the plot
            print(f"[DONE] {self.name} plotted trace")
        plt.close()

# ------------- plot posterior --------------- #
    def plot_posterior(self, var_names=None, figsize=None, save=True, show=False):
        print(f"[TASK] {self.name} plotting posterior")
        fig, ax = plt.subplots()
        with self.model:
            az.plot_posterior(self.trace, var_names=var_names, figsize=figsize)
        if save:
            plt.savefig(self.eval_path+'posterior.png')  # save the plot to a file
            print(f"[DONE] {self.name} saved posterior to file {self.eval_path}posterior.png")
        if show:
            plt.show()
            print(f"[DONE] {self.name} plotted posterior")
        plt.close()

# ----------- predictive checks ------------- #
    def prior_predictive(self,save=True,show=False):
        print(f"[TASK] {self.name} computing prior predictive check")
        with self.model:
            prior_predictive_samples = pm.sample_prior_predictive()
        # Extract the target variable predictions
        target_preds = prior_predictive_samples.prior_predictive.target.values.flatten()

        # Plot a histogram of the predicted revenue values
        fig, ax = plt.subplots()
        plt.hist(target_preds, bins=50, density=True)
        plt.xlabel('Predicted revenue')
        plt.ylabel('Density')
        plt.title('Prior predictive distribution of revenue')
        if save:
            plt.savefig(self.eval_path+'prior_predictive.png')
            print(f"[DONE] {self.name} saved prior predictive to file {self.eval_path}prior_predictive.png")
        if show:
            plt.show()
        plt.close()

    def generate_ppc(self):
        print(f"[TASK] {self.name} computing posterior predictive check")
        with self.model:
            self.ppc = pm.sample_posterior_predictive(self.trace)
        print(f"[DONE] {self.name} posterior predictive check computed ")

    def plot_ppc(self, save=True, show=False):
        print(f"[TASK] {self.name} plotting posterior predictive")
        fig, ax = plt.subplots()
        with self.model:
            az.plot_ppc(self.ppc)
            plt.xlim(-3,4)
        if save:
            plt.savefig(self.eval_path+'ppc.png')  # save the plot to a file
            print(f"[DONE] {self.name} saved ppc to file {self.eval_path}ppc.png")
        if show:
            plt.show()  # display the plot
            print(f"[DONE] {self.name} plotted ppc")
        plt.close()

# ----------- forecasting ------------- #
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

        print(f"[DONE] {self.name} test samples generated from predictive distribution ")
        return flat_predictions

    def plot_predictions(self,prediction_data,flat_predictions,true_values=True,ci=0.95,multiday=False):
        print(f"[TASK] {self.name} plotting predictions")

        PRED_COLOR = 'r'
        TRUE_COLOR = 'k'
        ACCENT = 'orange'

        fig, ax = plt.subplots()
        
        date_range = pd.date_range(prediction_data.date.min(), periods=len(prediction_data), freq='D')

        # plot hdi
        az.plot_hdi(date_range, flat_predictions, hdi_prob=ci, smooth=False,plot_kwargs={'alpha': 0.3},ax=ax)
        # az.plot_hdi(date_range, flat_predictions, hdi_prob=0.75, smooth=False,plot_kwargs={'alpha': 0.1},ax=ax)
        az.plot_hdi(date_range, flat_predictions, hdi_prob=0.5, smooth=False,plot_kwargs={'alpha': 0.3},ax=ax)
        ax.legend([f'{ci*100}% HDI'], loc='upper left', framealpha=1)

        # plot predicted mean
        plt.plot(date_range,flat_predictions.mean(axis=0),label='Predicted Mean',c=PRED_COLOR,linestyle='--')
        
        # rotate dates
        plt.xticks(rotation=45)

        if true_values:
            # plot true values
            plt.plot(date_range, prediction_data.revenue, label='True',c=TRUE_COLOR,linestyle='-',alpha=0.7)
            # plot error between true and predicted points
            # plt.vlines(date_range, prediction_data.revenue, flat_predictions.mean(axis=0), alpha=0.6,color=ACCENT)
            # plt.fill_between(date_range, prediction_data.revenue, flat_predictions.mean(axis=0), alpha=0.2, label='Error',color=ACCENT)

        # add a vertical line to indicate the end of the training data
        # plt.axvline(pd.to_datetime(self.test_data.date.min()), c='k', linestyle='--', alpha=0.3)
        # add text to indicate the end of the training data
        # plt.text(pd.to_datetime(self.test_data.date.min()), ax.get_ylim()[1]*0.7, 'Test Start', rotation=90, alpha=0.5)

        plt.legend()
        return fig

    def plot_predictions_v2(self,prediction_data, flat_predictions, true_values=True, ci=0.95):
        PRED_COLOR = 'r'
        TRUE_COLOR = 'k'
        ACCENT = 'orange'

        fig, ax = plt.subplots()
        
        date_range = pd.date_range(prediction_data.date.min(), periods=len(prediction_data), freq='D')

        # plot hdi
        az.plot_hdi(date_range, flat_predictions, hdi_prob=ci, smooth=False, plot_kwargs={'alpha': 0.3}, ax=ax)
        ax.legend([f'{ci*100}% HDI'], loc='upper left', framealpha=1)

        # plot predicted mean
        plt.plot(date_range, flat_predictions.mean(axis=0), label='Predicted Mean', c=PRED_COLOR, linestyle='--')
        
        # rotate dates
        plt.xticks(rotation=45)

        if true_values:
            # plot true values
            plt.plot(date_range, prediction_data.revenue, label='True', c=TRUE_COLOR, linestyle='-', alpha=0.7)
            # plot error between true and predicted points
            plt.vlines(date_range, prediction_data.revenue, flat_predictions.mean(axis=0), alpha=0.6, color=ACCENT)
            plt.fill_between(date_range, prediction_data.revenue, flat_predictions.mean(axis=0), alpha=0.2, label='Error', color=ACCENT)

        # KDE plot
        for idx, date in enumerate(date_range):
            kde = sns.kdeplot(flat_predictions[:, idx], bw_method='scott', color='purple', alpha=0.5, ax=ax)
            kde.set(xlim=(prediction_data.revenue.min(), prediction_data.revenue.max()))
            kde.lines[-1].set_label('Density' if idx == 0 else '')

        plt.legend()
        return fig

    def plot_pred_distribution(self,prediction_data,flat_predictions,true_values=True):
        print(f"[TASK] {self.name} plotting test distribution")

        PRED_COLOR = 'r'
        TRUE_COLOR = 'k'
        ACCENT = 'orange'

        fig, ax = plt.subplots()
        # plot all predictions as a histogram with a density plot 
        az.plot_dist(flat_predictions, kind='kde', rug=True, label='Predicted',color=PRED_COLOR)

        # add a vertical line for the mean
        plt.vlines(flat_predictions.mean(), *plt.gca().get_ylim(),color=PRED_COLOR,linestyle='--')

        if true_values:
            # overlay the observed data distribution
            az.plot_dist(prediction_data.revenue, kind='kde', rug=True, label='True',color=TRUE_COLOR)
            # add a vertical line for the mean
            plt.vlines(prediction_data.revenue.mean(), *plt.gca().get_ylim(),color=TRUE_COLOR,linestyle='--')

        plt.title('Posterior Predictive Distribution on Test')
        plt.legend(loc='upper left', framealpha=1)
        
        return fig

    def validate(self,save=True,show=False):
        print(f"[TASK] {self.name} validating model")
        
        # get predictions
        valid_predictions = self.generate_predictions(data=self.valid_data)
        fig = self.plot_predictions(prediction_data=self.valid_data,flat_predictions=valid_predictions,true_values=True, ci=0.95)

        if save:
            plt.savefig(self.eval_path+'valid/predictions.png')
            print(f"[DONE] {self.name} saved validation plot to file {self.eval_path}valid/predictions.png")
        if show:
            plt.show()
        plt.close()

        fig = self.plot_pred_distribution(self.valid_data,valid_predictions,)

        if save:
            plt.savefig(self.eval_path+'valid/prediction_distribtution.png')
            print(f"[DONE] {self.name} saved plot to file {self.eval_path}valid/prediction_distribution.png")
        if show:
            plt.show()
        plt.close()
        
        # compute metrics 
        valid_metrics = self.compute_metrics(valid_predictions,data=self.valid_data)

        with open(self.eval_path+'valid/valid_metrics.json', 'w') as f:
            json.dump(valid_metrics, f)
        print(f"[DONE] {self.name} saved validation metrics to file {self.eval_path}valid/valid_metrics.json")

    def test(self,save=True,show=False,multiday=False):
        print(f"[TASK] {self.name} testing model")

        # get prediction
        test_predictions = self.generate_predictions(data=self.test_data,multiday=multiday)
        fig = self.plot_predictions(self.test_data,test_predictions,true_values=True, ci=0.95)
        if save:
            plt.savefig(self.eval_path+'test/predictions.png')
            print(f"[DONE] {self.name} saved test plot to file {self.eval_path}test/predictions.png")
        if show:
            plt.show()
        plt.close()

        fig = self.plot_pred_distribution(self.test_data,test_predictions,)

        if save:
            plt.savefig(self.eval_path+'test/prediction_distribtution.png')
            print(f"[DONE] {self.name} saved test plot to file {self.eval_path}prediction_distribution.png")
        if show:
            plt.show()
        plt.close()

        # compute metrics
        test_metrics = self.compute_metrics(test_predictions,data=self.test_data)
        with open(self.eval_path+'test/test_metrics.json', 'w') as f:
            json.dump(test_metrics, f)
        print(f"[DONE] {self.name} saved test metrics to file {self.eval_path}test/test_metrics.json")

    def compute_metrics(self,predictions,data):
        print(f"[TASK] {self.name} computing metrics")

        true_data = data.revenue.to_numpy()
        predictions = np.array(predictions)

        # Calculate the metrics for each sample
        mae = np.mean(np.abs(predictions - true_data), axis=1)
        mse = np.mean(np.square(predictions - true_data), axis=1)
        rmse = np.sqrt(mse)
        mean_predictions = np.mean(predictions, axis=0)
        mad = np.mean(np.abs(mean_predictions - true_data))
        
        # Calculate log-likelihood
        log_likelihoods = np.zeros(predictions.shape[0])
        for i in range(predictions.shape[0]):
            log_likelihoods[i] = stats.norm.logpdf(true_data, loc=predictions[i], scale=np.std(predictions, axis=0)).sum()
        log_likelihood = np.mean(log_likelihoods)

        # Calculate the mean and standard deviation of the metrics across all samples
        metrics = {
            'MAE': {'mean': np.mean(mae), 'std': np.std(mae)},
            'MSE': {'mean': np.mean(mse), 'std': np.std(mse)},
            'RMSE': {'mean': np.mean(rmse), 'std': np.std(rmse)},
            'Log-likelihood': log_likelihood,
            'MAD': mad
        }

        return metrics

