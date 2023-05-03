import sys
sys.path.append('/Users/louisbrandt/itu/6/bachelor/models/classes')
import argparse
import pandas as pd
import arviz as az
from simple_temp import SimpleTempModel
from weatherglm import WeatherGLM
from weatherglmv2 import WeatherGLMv2
from laggedar import LaggedARModel
from combinedmodel import CombinedModel
from deploy import Deploy

def parse_arguments():
    parser = argparse.ArgumentParser(description='Bayesian workflow.')
    parser.add_argument('model', type=str, help='Model to use', choices=['weatherglm','weatherglmv2','simpletemp','laggedar','combinedmodel','deploy'])
    parser.add_argument('--n_lags', type=int, default=7, help='Number of lags to use in AR models')
    parser.add_argument('--n_days', type=int, default=90, help='Number of last days of data to use')
    parser.add_argument('--n_test_days', type=int, default=21, help='Number of test days')
    parser.add_argument('--draws', type=int, default=1000, help='Number of MCMC samples')
    parser.add_argument('--tune', type=int, default=1000, help='Number of tuning steps for NUTS sampler')
    parser.add_argument('--chains', type=int, default=4, help='Number of MCMC chains')
    parser.add_argument('--target-accept', type=float, default=0.9, help='Target acceptance rate for NUTS sampler')
    parser.add_argument('--test', action='store_true', help='Predict on test data or not')
    parser.add_argument('--sample-posterior', action='store_true', help='Sample posterior (otherwise, load trace)')
    parser.add_argument('--only-predict', action='store_true', help='After getting trace - only predict')
    parser.add_argument('--pred-ci', type=float, default=0.95, help='Confidence interval for prediction plots')
    parser.add_argument('--multiday', action='store_true', help='Forecast multiple days ahead, using predictions')
    return parser.parse_args()

def get_model_class_and_params(args):
    if args.model == 'weatherglm':
        model_class = WeatherGLM
        model_params = {'n_days': args.n_days}
    elif args.model == 'weatherglmv2':
        model_class = WeatherGLMv2
        model_params = {'n_days': args.n_days}
    elif args.model == 'simpletemp':
        model_class = SimpleTempModel
        model_params = {'n_days': args.n_days}
    elif args.model == 'laggedar':
        model_class = LaggedARModel
        model_params = {'n_days': args.n_days,'n_lags': args.n_lags}
    elif args.model == 'combinedmodel':
        model_class = CombinedModel
        model_params = {'n_days': args.n_days,'n_lags': args.n_lags}
    elif args.model == 'deploy':
        model_class = Deploy
        model_params = {'n_days': args.n_days,'n_lags': args.n_lags}
    return model_class, model_params

def main():
    args = parse_arguments()
    model_class, model_params = get_model_class_and_params(args)

    # instantiate model
    model = model_class(**model_params)

    # sample posterior or load trace
    if args.sample_posterior:
        model.sample_posterior(draws=args.draws, tune=args.tune, chains=args.chains, target_accept=args.target_accept)
        model.save_trace()
    else:
        model.load_trace()
    
    if not args.only_predict:
    # --- priorpc
        model.prior_predictive()

    # --- print summary
        model.summary()

    # --- plot trace
        model.plot_trace()
        
    # --- ppc
        model.generate_ppc()
        model.plot_ppc()

# --- validate
    model.validate()

    if args.test:
    # --- predictions
        model.test(multiday=args.multiday)

if __name__ == '__main__':
    main()
