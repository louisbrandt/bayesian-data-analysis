import sys
sys.path.append('/Users/louisbrandt/itu/6/bachelor/models/classes')
import argparse
import pandas as pd
import arviz as az
from simple_temp import SimpleTempModel
from weatherglm import WeatherGLM
from naivear import ARModel
from laggedar import LaggedARModel

def parse_arguments():
    parser = argparse.ArgumentParser(description='Bayesian workflow.')
    parser.add_argument('model', type=str, help='Model to use', choices=['weatherglm', 'simpletempmodel', 'armodel','laggedarmodel'])
    parser.add_argument('--data-path', type=str, default='/Users/louisbrandt/itu/6/bachelor/data/processed_data.csv', help='Path to data file')
    parser.add_argument('--n_days', type=int, default=90, help='Number of last days of data to use')
    parser.add_argument('--n_test_days', type=int, default=21, help='Number of test days')
    parser.add_argument('--draws', type=int, default=1000, help='Number of MCMC samples')
    parser.add_argument('--tune', type=int, default=1000, help='Number of tuning steps for NUTS sampler')
    parser.add_argument('--chains', type=int, default=4, help='Number of MCMC chains')
    parser.add_argument('--target-accept', type=float, default=0.9, help='Target acceptance rate for NUTS sampler')
    parser.add_argument('--test', action='store_true', help='Predict on test data or not')
    parser.add_argument('--sample-posterior', action='store_true', help='Sample posterior (otherwise, load trace)')
    parser.add_argument('--include-recent', action='store_true', help='Include last 15 days in predictions')
    parser.add_argument('--only-predict', action='store_true', help='After getting trace - only predict')
    parser.add_argument('--pred-ci', type=float, default=0.95, help='Confidence interval for prediction plots')
    return parser.parse_args()

def get_model_class_and_params(args):
    if args.model == 'weatherglm':
        model_class = WeatherGLM
        model_params = {'data_path': args.data_path, 'n_test_days': args.n_test_days, 'n_days': args.n_days}
    elif args.model == 'armodel':
        model_class = ARModel
        model_params = {'data_path': args.data_path, 'n_test_days': args.n_test_days, 'n_days': args.n_days}
    elif args.model == 'simpletempmodel':
        model_class = SimpleTempModel
        model_params = {'data_path': args.data_path, 'n_test_days': args.n_test_days, 'n_days': args.n_days}
    elif args.model == 'laggedarmodel':
        model_class = LaggedARModel
        model_params = {'data_path': args.data_path, 'n_test_days': args.n_test_days, 'n_days': args.n_days}
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

    if args.test:
    # --- predictions
        model.generate_predictions(include_recent=args.include_recent)
        model.plot_test_distribution()
        model.plot_predictions()
        model.report_model()

if __name__ == '__main__':
    main()
