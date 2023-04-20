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

# ----------- Parent Class ------------- #
class BayesianModel():
    def __init__(self,name,data_path,cat_cols,num_cols,n_days,n_test_days):
        self.name = name
        self.data = None
        self.test_data = None
        self.trace = None
        self.posterior_predictive = None
        self.data_path = data_path
        self.cat_cols = cat_cols if cat_cols is not None else []
        self.num_cols = num_cols if num_cols is not None else []
        self.time_index_flag=False
        self.n_days = n_days
        self.n_test_days = n_test_days
        self.set_data()
        self.split_data()
        self.define_model()
        # create a dictionary of the hyperparameters to store with the trace and plots for reproducibility
        self.hyperparams = {
                'data': str(self.n_days) + '_days',
                'test': str(self.n_test_days) + '_days_from' + str(self.test_data.date.min())
            }
        # add hyperparams to the pickle & plot path at the end of the string
        self.pkl_path = '/Users/louisbrandt/itu/6/bachelor/pickles/'+self.name+'/'
        self.eval_path = '/Users/louisbrandt/itu/6/bachelor/eval/'+self.name+'/'
        self.pkl_path += f"data={self.hyperparams['data']}"
        self.eval_path += '&'.join([f'{key}={val}' for key, val in self.hyperparams.items()])
        self.pkl_path += '/'
        self.eval_path += '/'
        # Check if directories exist and create them if not
        for path in [self.pkl_path, self.eval_path]:
            if not os.path.exists(path):
                os.makedirs(path)

    def split_data(self):
        self.test_data = self.data[-self.n_test_days:]
        self.data = self.data[:-self.n_test_days]
        self.data = self.data[-self.n_days:]
    
    def set_data(self):
        data = pd.read_csv(self.data_path)
        self.data = data.astype({col: 'category' for col in self.cat_cols})
    
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
            cloudpickle.dump({'model': self.model, 'trace': self.trace,'y_obs':self.data}, f)
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
    def generate_predictions(self,include_recent=False):
        print(f"[TASK] {self.name} generating samples from predictive distribution given data")

        self.prediction_data = self.test_data.reset_index(drop=True)

        with self.model:
            # set data to test data in order to set the mean of the likelihood distribution
            for var in self.cat_cols:
                pm.set_data({var: self.prediction_data[var].cat.codes})
            for var in self.num_cols:
                pm.set_data({var: self.prediction_data[var].to_numpy()})

            pm.set_data({'revenue': self.prediction_data['revenue'].to_numpy()}) # for shape

            # generate samples from predictive distribution
            self.test_posterior_predictive = pm.sample_posterior_predictive(self.trace, var_names=['target'])
            self.predictions = self.test_posterior_predictive.posterior_predictive.target.values

            if include_recent:
                # include last 15 days of test predictions in plot
                self.prediction_data = pd.concat([self.data[-15:], self.prediction_data],ignore_index=True)

                # get last 15 days of training predictions
                ppc = self.ppc.posterior_predictive.target.values
                # flatten ppc
                flat_ppc = ppc.reshape(-1, ppc.shape[-1])
                # get last 15 days of training predictions
                flat_ppc = flat_ppc[:,-15:]
                flat_predictions = self.predictions.reshape(-1, self.predictions.shape[-1])
                self.flat_predictions = np.concatenate([flat_ppc, flat_predictions], axis=1)
            else:
                self.flat_predictions = self.predictions.reshape(-1, self.predictions.shape[-1])

        print(f"[DONE] {self.name} test samples generated from predictive distribution ")

    def plot_test_distribution(self,true_values=True, save=True, show=False):
        print(f"[TASK] {self.name} plotting test distribution")

        PRED_COLOR = 'r'
        TRUE_COLOR = 'k'
        ACCENT = 'orange'

        fig, ax = plt.subplots()
        # plot all predictions as a histogram with a density plot 
        az.plot_dist(self.flat_predictions, kind='kde', rug=True, label='Predicted',color=PRED_COLOR)

        # add a vertical line for the mean
        plt.vlines(self.flat_predictions.mean(), *plt.gca().get_ylim(),color=PRED_COLOR,linestyle='--')

        if true_values:
            # overlay the observed data distribution
            az.plot_dist(self.test_data.revenue, kind='kde', rug=True, label='True',color=TRUE_COLOR)
            # add a vertical line for the mean
            plt.vlines(self.test_data.revenue.mean(), *plt.gca().get_ylim(),color=TRUE_COLOR,linestyle='--')

        plt.title('Posterior Predictive Distribution on Test')
        plt.legend(loc='upper left', framealpha=1)

        if save:
            plt.savefig(self.eval_path+'prediction_distribution.png')  # save the plot to a file
            print(f"[DONE] {self.name} saved prediction distribution to file {self.eval_path}prediction_distribution.png")
        if show:
            plt.show()
        plt.close()


    def plot_predictions(self, true_values=True, save=True, show=False, distplot=True, ci=0.95):
        print(f"[TASK] {self.name} plotting predictions")

        PRED_COLOR = 'r'
        TRUE_COLOR = 'k'
        ACCENT = 'orange'

        fig, ax = plt.subplots()
        
        date_range = pd.date_range(self.prediction_data.date.min(), periods=len(self.prediction_data), freq='D')

        # plot hdi
        az.plot_hdi(date_range, self.flat_predictions, hdi_prob=ci, smooth=False,plot_kwargs={'alpha': 0.3},ax=ax)
        ax.legend([f'{ci*100}% HDI'], loc='upper left', framealpha=1)

        # plot predicted mean
        plt.plot(date_range,self.flat_predictions.mean(axis=0),label='Predicted Mean',c=PRED_COLOR,linestyle='--')
        
        # rotate dates
        plt.xticks(rotation=45)

        if true_values:
            # plot true values
            plt.plot(date_range, self.prediction_data.revenue, label='True',c=TRUE_COLOR,linestyle='-',alpha=0.7)
            # plot error between true and predicted points
            plt.vlines(date_range, self.prediction_data.revenue, self.flat_predictions.mean(axis=0), alpha=0.6,color=ACCENT)
            plt.fill_between(date_range, self.prediction_data.revenue, self.flat_predictions.mean(axis=0), alpha=0.2, label='Error',color=ACCENT)

        # add a vertical line to indicate the end of the training data
        plt.axvline(pd.to_datetime(self.test_data.date.min()), c='k', linestyle='--', alpha=0.3)
        # add text to indicate the end of the training data
        plt.text(pd.to_datetime(self.test_data.date.min()), ax.get_ylim()[1]*0.7, 'Test Start', rotation=90, alpha=0.5)

        plt.legend()
        if save:
            plt.savefig(self.eval_path+'predictions.png')  # save the plot to a file
            print(f"[DONE] {self.name} saved predictions to file {self.eval_path}predictions.png")
        if show:
            plt.show()
        plt.close()

    def compute_log_likelihood(self):
        # Compute the standard deviation of the predictive distribution for each observation
        predictive_std = np.std(self.flat_predictions, axis=0)
        # Compute the mean of the predictive distribution for each observation
        predictive_mean = np.mean(self.flat_predictions, axis=0)
        # Compute the log-likelihood of the predictions
        log_likelihoods = stats.norm.logpdf(self.prediction_data['revenue'].to_numpy(), loc=predictive_mean, scale=predictive_std)

        return np.sum(log_likelihoods)

    def report_model(self):
        # Compute log likelihood
        log_like = self.compute_log_likelihood()

        # Prepare the output string
        output_string = f"{self.name}{self.n_days} Model Metrics:\n"
        output_string += f"log likelihood: {log_like}\n"

        # Write the output string to a file
        with open(self.eval_path+'model_metrics.txt', 'w') as f:
            f.write(output_string)
            f.write('\n')

        print('[DONE]',output_string)

