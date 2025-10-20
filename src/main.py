import argparse
import logging
import os
import signal
import sys
from datetime import date, datetime

import torch
from functions.LBFGS import FullBatchLBFGS
from set_seed import set_seed
from torch.optim import LBFGS, Adam, AdamW
from trainers.base_trainer import BaseTrainer
from trainers.ca_gp_trainer import (
    FexoLogEICaGPSlidingWindowTrainer, HartmannEICaGPEULBOTrainer,
    HartmannEICaGPTrainer, HartmannLogEICaGPTrainer,
    LassoDNALogEICaGPSlidingWindowTrainer, LassoDNALogEICaGPTrainer,
    LunarEICaGPEULBOTrainer, LunarEICaGPTrainer, LunarLogEICaGPEULBOTrainer,
    LunarLogEICaGPTrainer, Med1LogEICaGPSlidingWindowTrainer,
    Med2LogEICaGPSlidingWindowTrainer, OsmbLogEICaGPSlidingWindowTrainer,
    OsmbLogEICaGPTrainer, RoverEICaGPEULBOTrainer,
    RoverEICaGPSlidingWindowTrainer, RoverEICaGPTrainer,
    PdopLogEICaGPSlidingWindowTrainer, AdipLogEICaGPSlidingWindowTrainer,
    RanoLogEICaGPSlidingWindowTrainer)
from trainers.exact_gp_trainer import (
    FexoLogEIExactGPSlidingWindowTrainer, FexoLogEIExactGPTrainer,
    FexoLogEIGPyTorchExactGPSlidingWindowTrainer, HartmannEIExactGPTrainer,
    LassoDNALogEIExactGPSlidingWindowTrainer, LassoDNALogEIExactGPTrainer,
    LunarEIExactGPTrainer, Med1LogEIExactGPSlidingWindowTrainer,
    Med1LogEIExactGPTrainer, Med2LogEIExactGPSlidingWindowTrainer,
    Med2LogEIExactGPTrainer, OsmbLogEIExactGPSlidingWindowTrainer,
    OsmbLogEIExactGPTrainer, RoverEIExactGPSlidingWindowTrainer,
    RoverEIExactGPTrainer)
from trainers.sgpr_trainer import FexoLogEISGPRTrainer, OsmbLogEISGPRTrainer
from trainers.svgp_trainer import (
    FexoLogEISVGPTrainer, HartmannEISVGPEULBOTrainer,
    HartmannEISVGPRetrainTrainer, HartmannEISVGPTrainer,
    LassoDNALogEISVGPTrainer, LunarEISVGPEULBOTrainer, LunarEISVGPTrainer,
    Med1LogEISVGPTrainer, Med2LogEISVGPTrainer, OsmbLogEISVGPTrainer,
    RoverEISVGPEULBOTrainer, RoverEISVGPTrainer, PdopLogEISVGPTrainer,
    AdipLogEISVGPTrainer, RanoLogEISVGPTrainer)

import wandb

arg_trainer_map = {
    'hartmann_ei_exact_gp': HartmannEIExactGPTrainer,
    'hartmann_ei_svgp': HartmannEISVGPTrainer,
    'hartmann_ei_svgp_eulbo': HartmannEISVGPEULBOTrainer,
    'hartmann_ei_svgp_retrain': HartmannEISVGPRetrainTrainer,
    'hartmann_ei_ca_gp': HartmannEICaGPTrainer,
    'hartmann_ei_ca_gp_eulbo': HartmannEICaGPEULBOTrainer,
    'hartmann_log_ei_ca_gp': HartmannLogEICaGPTrainer,
    'lunar_ei_exact_gp': LunarEIExactGPTrainer,
    'lunar_ei_ca_gp': LunarEICaGPTrainer,
    'lunar_log_ei_ca_gp': LunarLogEICaGPTrainer,
    'lunar_ei_svgp': LunarEISVGPTrainer,
    'lunar_ei_svgp_eulbo': LunarEISVGPEULBOTrainer,
    'lunar_ei_ca_gp_eulbo': LunarEICaGPEULBOTrainer,
    'lunar_log_ei_ca_gp_eulbo': LunarLogEICaGPEULBOTrainer,
    'rover_ei_exact_gp': RoverEIExactGPTrainer,
    'rover_ei_exact_gp_sliding_window': RoverEIExactGPSlidingWindowTrainer,
    'rover_ei_ca_gp': RoverEICaGPTrainer,
    'rover_ei_ca_gp_eulbo': RoverEICaGPEULBOTrainer,
    'rover_ei_ca_gp_sliding_window': RoverEICaGPSlidingWindowTrainer,
    'rover_ei_svgp': RoverEISVGPTrainer,
    'rover_ei_svgp_eulbo': RoverEISVGPEULBOTrainer,
    'lasso_dna_log_ei_exact_gp': LassoDNALogEIExactGPTrainer,
    'lasso_dna_log_ei_exact_gp_sliding_window':
    LassoDNALogEIExactGPSlidingWindowTrainer,
    'lasso_dna_log_ei_ca_gp': LassoDNALogEICaGPTrainer,
    'lasso_dna_log_ei_ca_gp_sliding_window':
    LassoDNALogEICaGPSlidingWindowTrainer,
    'lasso_dna_log_ei_svgp': LassoDNALogEISVGPTrainer,
    'osmb_log_ei_exact_gp': OsmbLogEIExactGPTrainer,
    'osmb_log_ei_exact_gp_sliding_window':
    OsmbLogEIExactGPSlidingWindowTrainer,
    'osmb_log_ei_ca_gp': OsmbLogEICaGPTrainer,
    'osmb_log_ei_ca_gp_sliding_window': OsmbLogEICaGPSlidingWindowTrainer,
    'osmb_log_ei_svgp': OsmbLogEISVGPTrainer,
    'osmb_log_ei_sgpr': OsmbLogEISGPRTrainer,
    'fexo_log_ei_exact_gp': FexoLogEIExactGPTrainer,
    'fexo_log_ei_exact_gp_sliding_window':
    FexoLogEIExactGPSlidingWindowTrainer,
    'fexo_log_ei_gpytorch_exact_gp_sliding_window':
    FexoLogEIGPyTorchExactGPSlidingWindowTrainer,
    'fexo_log_ei_ca_gp_sliding_window': FexoLogEICaGPSlidingWindowTrainer,
    'fexo_log_ei_svgp': FexoLogEISVGPTrainer,
    'fexo_log_ei_sgpr': FexoLogEISGPRTrainer,
    'med1_log_ei_exact_gp': Med1LogEIExactGPTrainer,
    'med1_log_ei_exact_gp_sliding_window':
    Med1LogEIExactGPSlidingWindowTrainer,
    'med1_log_ei_ca_gp_sliding_window': Med1LogEICaGPSlidingWindowTrainer,
    'med1_log_ei_svgp': Med1LogEISVGPTrainer,
    'med2_log_ei_exact_gp': Med2LogEIExactGPTrainer,
    'med2_log_ei_exact_gp_sliding_window':
    Med2LogEIExactGPSlidingWindowTrainer,
    'med2_log_ei_ca_gp_sliding_window': Med2LogEICaGPSlidingWindowTrainer,
    'med2_log_ei_svgp': Med2LogEISVGPTrainer,
    'pdop_log_ei_ca_gp_sliding_window': PdopLogEICaGPSlidingWindowTrainer,
    'pdop_log_ei_svgp': PdopLogEISVGPTrainer,
    'adip_log_ei_ca_gp_sliding_window': AdipLogEICaGPSlidingWindowTrainer,
    'adip_log_ei_svgp': AdipLogEISVGPTrainer,
    'rano_log_ei_ca_gp_sliding_window': RanoLogEICaGPSlidingWindowTrainer,
    'rano_log_ei_svgp': RanoLogEISVGPTrainer,
}
arg_optimizer_map = {
    'adamW': AdamW,
    'adam': Adam,
    'lbfgs': LBFGS,
    'custom_lbfgs': FullBatchLBFGS,
    'botorch_lbfgs': 'botorch_lbfgs'
}

# Map string names to torch dtypes
TORCH_DTYPES = {
    'float32': torch.float32,
    'float64': torch.float64,
}


def handler(self, signum, frame):
    # if we Ctrl-c, make sure we terminate wandb tracker
    print('Ctrl-c hass been pressed, terminating wandb tracker...')
    self.tracker.finish()
    msg = 'tracker terminated, now exiting...'
    print(msg, end='', flush=True)
    exit(1)


def parse_dtype(dtype_str):
    if dtype_str in TORCH_DTYPES:
        return TORCH_DTYPES[dtype_str]
    raise argparse.ArgumentTypeError(
        f"Invalid dtype: '{dtype_str}'. Choose from {list(TORCH_DTYPES.keys())}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Run computation aware GP based BO')

    parser.add_argument('--epochs',
                        default=30,
                        type=int,
                        help='number of epochs to train model')
    parser.add_argument('--eulbo_epochs',
                        default=30,
                        type=int,
                        help='number of epochs to train model')
    parser.add_argument('--device',
                        '-d',
                        default='cuda',
                        type=str,
                        help='cpu or gpu ID to use')
    parser.add_argument('--batch_size',
                        default=1,
                        type=int,
                        help='acquisition batch size')
    parser.add_argument('--dropout_prob',
                        default=0.3,
                        type=float,
                        help='probability for dropout layers')
    parser.add_argument('--grad_clip',
                        default=2.0,
                        type=float,
                        help='probability for dropout layers')
    parser.add_argument('--save_dir', help='path to saved model files')
    parser.add_argument('--data_dir', help='path to data files')
    parser.add_argument('--optimizer',
                        default='adam',
                        help='type of optimizer to use')
    parser.add_argument('--num_repeats',
                        default=3,
                        type=int,
                        help='number of times to repeat experiment')
    parser.add_argument('--seed',
                        default=11202022,
                        type=int,
                        help='random seed to be used in numpy and torch')
    parser.add_argument('--learning_rate',
                        default=1e-2,
                        type=float,
                        help='learning rate for optimizer')
    parser.add_argument(
        '--ca_gp_actions_learning_rate',
        default=1e-2,
        type=float,
        help='learning rate for CaGP action specific optimizer')
    parser.add_argument(
        '--svgp_inducing_point_learning_rate',
        default=1e-2,
        type=float,
        help='learning rate for SVGP inducing point specific optimizer')
    parser.add_argument('--max_oracle_calls',
                        default=2000,
                        type=int,
                        help='max number of function evals/oracle calls')
    parser.add_argument('--trainer_type',
                        default='lunar_ei_ca_gp',
                        help='type of experiment to run')
    parser.add_argument('--kernel_type',
                        default='matern_5_2',
                        help='kernel type for GP')
    parser.add_argument('--kernel_likelihood_prior',
                        default='none',
                        help='kernel and likelihood prior for GP')
    parser.add_argument(
        '--use_ard_kernel',
        action='store_true',
        help='fit a separate lengthscale for each input dimension')
    parser.add_argument(
        '--turn_on_outcome_transform',
        action='store_true',
        help='turn on standardize outcome transform for non exact GPs')
    parser.add_argument('--turn_on_input_transform',
                        action='store_true',
                        help='turn on normalize transform for non exact GPs')
    parser.add_argument('--ca_gp_init_mode',
                        default='random',
                        help='init mode for ca gp')
    parser.add_argument('--norm_data',
                        action='store_true',
                        help='normalize ys')
    parser.add_argument(
        '--reinit_hyperparams',
        action='store_true',
        help='reinitialize kernel hyperparameters at each step')
    parser.add_argument('--roll_actions',
                        action='store_true',
                        help='platform/roll actions')
    parser.add_argument('--reinit_mean',
                        action='store_true',
                        help='reinitialize mean hyperparameters at each step')
    parser.add_argument(
        '--log_diagnostics',
        action='store_true',
        help='log diagnostic metrics, will slow down runs slightly')
    parser.add_argument('--freeze_actions',
                        action='store_true',
                        help='freeze actions to be unit norm, for debugging')
    parser.add_argument(
        '--non_zero_action_init',
        action='store_true',
        help='initialize CaGP actions uniformly, away from zero')
    parser.add_argument('--add_actions_by_reinit',
                        action='store_true',
                        help='reinitialize actions when adding')
    parser.add_argument('--turn_off_wandb',
                        action='store_true',
                        help='skip wandb logging')
    parser.add_argument('--add_likelihood_to_posterior',
                        action='store_true',
                        help='add likelihood to posterior for EI')
    parser.add_argument('--notes', default='', help='note on experiment run')
    parser.add_argument('--use_analytic_acq_func',
                        action='store_true',
                        help='use analytic acquisition function instead of MC')
    parser.add_argument('--enable_raasp',
                        action='store_true',
                        help='enable RAASP sampling in AF optimization')
    parser.add_argument('--raasp_best_pct',
                        default=5,
                        type=float,
                        help='pct of best points to perturb in RAASP')
    parser.add_argument('--raasp_sigma',
                        default=1e-3,
                        type=float,
                        help='std of perturbations in RAASP')
    parser.add_argument('--early_stopping_threshold',
                        default=3,
                        type=int,
                        help='patience for early stopping')
    parser.add_argument('--num_initial_points',
                        default=100,
                        type=int,
                        help='initial number of points to train model on')
    parser.add_argument('--update_train_size',
                        default=100,
                        type=int,
                        help='size of sliding window to update on')
    parser.add_argument('--num_inducing_points',
                        default=100,
                        type=int,
                        help='number of inducing points for svgp')
    parser.add_argument('--proj_dim_ratio',
                        default=0.5,
                        type=float,
                        help='ratio for ca gp projection dim')
    parser.add_argument(
        '--static_proj_dim',
        default=-1,
        type=int,
        help='if not -1, keep ca gp projection dim constant throughout training'
    )
    parser.add_argument(
        '--data_type',
        default='float64',
        type=parse_dtype,
        help=
        'Specify the PyTorch data type. Choose from: float16, float32, float64, etc. (default: float32)'
    )

    args = parser.parse_args()
    configs = args.__dict__
    configs['date'] = datetime.now().strftime('%m_%d_%Y_%H:%M:%S')

    # wandb tracking
    if configs['turn_off_wandb']:
        tracker = None
    else:
        tracker = wandb.init(project='Computation-Aware-BO',
                             group=configs['trainer_type'],
                             config=configs,
                             notes=configs['notes'])
        os.environ['WANDB_RUN_GROUP'] = 'experiment-' + configs['trainer_type']

    # for repeatability
    set_seed(configs['seed'])

    torch.set_default_dtype(configs['data_type'])
    # set default device for CaGP
    torch.set_default_device(torch.device(configs['device']))

    # set up logging
    filename = f'{configs["trainer_type"]}-{date.today()}'
    FORMAT = '%(asctime)s;%(levelname)s;%(message)s'
    logging.basicConfig(level=logging.INFO,
                        filename=f'{configs["save_dir"]}logs/{filename}.log',
                        filemode='a',
                        format=FORMAT)
    logging.info(configs)

    # get trainer
    trainer_type = arg_trainer_map[configs['trainer_type']]
    trainer: BaseTrainer = trainer_type(
        optimizer_type=arg_optimizer_map[configs['optimizer']],
        tracker=tracker,
        **configs)

    signal.signal(signal.SIGINT, handler)
    # perform experiment n times
    for iter in range(configs['num_repeats']):
        # for repeatability
        trainer.run_experiment(iter)
        trainer.task.num_calls = 0
        trainer.reinitialize_task()
        set_seed(configs['seed'] + iter + 1)
        if configs['turn_off_wandb']:
            continue

        # reinitialize tracker for each new run
        if iter != configs['num_repeats'] - 1:
            trainer.tracker.finish()
            tracker = wandb.init(project='Computation-Aware-BO',
                                 group=configs['trainer_type'],
                                 config=configs,
                                 notes=configs['notes'])
            trainer.init_new_run(tracker)

    return 0


if __name__ == '__main__':
    sys.exit(main())
