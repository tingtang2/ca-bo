import argparse
import logging
import random
import sys
from datetime import date

import torch
from torch import nn
from torch.optim import Adam, AdamW

from trainers.exact_gp_trainer import HartmannEIExactGPTrainer
from trainers.svgp_trainer import HartmannEISVGPTrainer

import wandb
import os
from datetime import datetime

arg_trainer_map = {
    'hartmann_ei_exact_gp': HartmannEIExactGPTrainer,
    'hartmann_ei_svgp': HartmannEISVGPTrainer
}
arg_optimizer_map = {'adamW': AdamW, 'adam': Adam}


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Run computation aware GP based BO')

    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        help='number of epochs to train model')
    parser.add_argument('--device',
                        '-d',
                        default='cpu',
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
    parser.add_argument('--save_dir', help='path to saved model files')
    parser.add_argument('--data_dir', help='path to data files')
    parser.add_argument('--optimizer',
                        default='adamW',
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
                        default=1e-3,
                        type=float,
                        help='learning rate for optimizer')
    parser.add_argument('--max_oracle_calls',
                        default=2000,
                        type=int,
                        help='max number of function evals/oracle calls')
    parser.add_argument('--trainer_type',
                        default='hartmann_ei_exact_gp',
                        help='type of experiment to run')
    parser.add_argument('--norm_data',
                        action='store_true',
                        help='normalize ys')
    parser.add_argument('--turn_off_wandb',
                        action='store_true',
                        help='skip wandb logging')

    args = parser.parse_args()
    configs = args.__dict__
    configs['date'] = datetime.now().strftime("%m/%d/%Y_%H:%M:%S")

    # wandb tracking
    if configs['turn_off_wandb']:
        tracker = None
    else:
        tracker = wandb.init(project='Computation-Aware-BO',
                             group=configs["trainer_type"],
                             config=configs)
        os.environ["WANDB_RUN_GROUP"] = "experiment-" + configs["trainer_type"]

    # for repeatability
    torch.manual_seed(configs['seed'])
    random.seed(configs['seed'])

    # need this precision for GP fitting
    torch.set_default_dtype(torch.float64)

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
    trainer = trainer_type(
        optimizer_type=arg_optimizer_map[configs['optimizer']],
        criterion=nn.CrossEntropyLoss(reduction='sum'),
        tracker=tracker,
        **configs)

    # perform experiment n times
    for iter in range(configs['num_repeats']):
        trainer.run_experiment(iter)

    return 0


if __name__ == '__main__':
    sys.exit(main())
