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
    AdipLogEICaGPSlidingWindowTrainer, AdipLogEICaGPTurboSlidingWindowTrainer,
    AdipLogEICaGPTurboTrainer, DhopLogEICaGPSlidingWindowTrainer,
    FexoLogEICaGPSlidingWindowTrainer, FexoLogEICaGPTurboSlidingWindowTrainer,
    FexoLogEICaGPTurboTrainer, HartmannEICaGPEULBOTrainer,
    HartmannEICaGPTrainer, HartmannEICaGPTurboSlidingWindowTrainer,
    HartmannEICaGPTurboTrainer, HartmannLogEICaGPTrainer,
    LassoDNALogEICaGPSlidingWindowTrainer,
    LassoDNALogEICaGPTurboSlidingWindowTrainer,
    LassoDNALogEICaGPTurboTrainer, LassoDNALogEICaGPTrainer,
    LunarEICaGPEULBOTrainer, LunarEICaGPTrainer,
    LunarEICaGPTurboSlidingWindowTrainer, LunarEICaGPTurboTrainer,
    LunarLogEICaGPEULBOTrainer, LunarLogEICaGPTrainer,
    Med1LogEICaGPSlidingWindowTrainer, Med1LogEICaGPTurboSlidingWindowTrainer,
    Med1LogEICaGPTurboTrainer, Med2LogEICaGPSlidingWindowTrainer,
    Med2LogEICaGPTurboSlidingWindowTrainer, Med2LogEICaGPTurboTrainer,
    OsmbLogEICaGPSlidingWindowTrainer, OsmbLogEICaGPTurboSlidingWindowTrainer,
    OsmbLogEICaGPTurboTrainer, OsmbLogEICaGPTrainer,
    PdopLogEICaGPSlidingWindowTrainer, PdopLogEICaGPTurboSlidingWindowTrainer,
    PdopLogEICaGPTurboTrainer, RanoLogEICaGPSlidingWindowTrainer,
    RanoLogEICaGPTurboSlidingWindowTrainer, RanoLogEICaGPTurboTrainer,
    RoverEICaGPEULBOTrainer, RoverEICaGPSlidingWindowTrainer,
    RoverEICaGPTurboSlidingWindowTrainer, RoverEICaGPTurboTrainer,
    RoverEICaGPTrainer, RoverLogEICaGPSlidingWindowTrainer,
    RoverLogEICaGPTurboSlidingWindowTrainer, RoverLogEICaGPTurboTrainer,
    ShopLogEICaGPSlidingWindowTrainer, SigaLogEICaGPSlidingWindowTrainer,
    ValtLogEICaGPSlidingWindowTrainer, ZaleLogEICaGPSlidingWindowTrainer)
from trainers.exact_gp_trainer import (
    AdipLogEIExactGPSlidingWindowTrainer, AdipLogEIExactGPTrainer,
    AdipLogEIExactGPTurboSlidingWindowTrainer, AdipLogEIExactGPTurboTrainer,
    FexoLogEIExactGPSlidingWindowTrainer, FexoLogEIExactGPTrainer,
    FexoLogEIExactGPTurboSlidingWindowTrainer, FexoLogEIExactGPTurboTrainer,
    FexoLogEIGPyTorchExactGPSlidingWindowTrainer, HartmannEIExactGPTrainer,
    HartmannEIExactGPTurboSlidingWindowTrainer, HartmannEIExactGPTurboTrainer,
    LassoDNALogEIExactGPSlidingWindowTrainer, LassoDNALogEIExactGPTrainer,
    LassoDNALogEIExactGPTurboSlidingWindowTrainer,
    LassoDNALogEIExactGPTurboTrainer, LunarEIExactGPTrainer,
    LunarEIExactGPTurboSlidingWindowTrainer, LunarEIExactGPTurboTrainer,
    Med1LogEIExactGPSlidingWindowTrainer, Med1LogEIExactGPTrainer,
    Med1LogEIExactGPTurboSlidingWindowTrainer, Med1LogEIExactGPTurboTrainer,
    Med2LogEIExactGPSlidingWindowTrainer, Med2LogEIExactGPTrainer,
    Med2LogEIExactGPTurboSlidingWindowTrainer, Med2LogEIExactGPTurboTrainer,
    OsmbLogEIExactGPSlidingWindowTrainer, OsmbLogEIExactGPTrainer,
    OsmbLogEIExactGPTurboSlidingWindowTrainer, OsmbLogEIExactGPTurboTrainer,
    PdopLogEIExactGPSlidingWindowTrainer, PdopLogEIExactGPTrainer,
    PdopLogEIExactGPTurboSlidingWindowTrainer, PdopLogEIExactGPTurboTrainer,
    RanoLogEIExactGPSlidingWindowTrainer, RanoLogEIExactGPTrainer,
    RanoLogEIExactGPTurboSlidingWindowTrainer, RanoLogEIExactGPTurboTrainer,
    RoverEIExactGPSlidingWindowTrainer, RoverEIExactGPTrainer,
    RoverEIExactGPTurboSlidingWindowTrainer, RoverEIExactGPTurboTrainer,
    RoverLogEIExactGPSlidingWindowTrainer, RoverLogEIExactGPTrainer,
    RoverLogEIExactGPTurboSlidingWindowTrainer,
    RoverLogEIExactGPTurboTrainer)
from trainers.sgpr_trainer import (
    AdipLogEISGPRTrainer, AdipLogEISGPRTurboSlidingWindowTrainer,
    AdipLogEISGPRTurboTrainer, DhopLogEISGPRTrainer, FexoLogEISGPRTrainer,
    FexoLogEISGPRTurboSlidingWindowTrainer, FexoLogEISGPRTurboTrainer,
    HartmannEISGPRTrainer, HartmannEISGPRTurboSlidingWindowTrainer,
    HartmannEISGPRTurboTrainer, LassoDNALogEISGPRTrainer,
    LassoDNALogEISGPRTurboSlidingWindowTrainer,
    LassoDNALogEISGPRTurboTrainer, LunarEISGPRTrainer,
    LunarEISGPRTurboSlidingWindowTrainer, LunarEISGPRTurboTrainer,
    Med1LogEISGPRTrainer, Med1LogEISGPRTurboSlidingWindowTrainer,
    Med1LogEISGPRTurboTrainer, Med2LogEISGPRTrainer,
    Med2LogEISGPRTurboSlidingWindowTrainer, Med2LogEISGPRTurboTrainer,
    OsmbLogEISGPRTrainer, OsmbLogEISGPRTurboSlidingWindowTrainer,
    OsmbLogEISGPRTurboTrainer, PdopLogEISGPRTrainer,
    PdopLogEISGPRTurboSlidingWindowTrainer, PdopLogEISGPRTurboTrainer,
    RanoLogEISGPRTrainer, RanoLogEISGPRTurboSlidingWindowTrainer,
    RanoLogEISGPRTurboTrainer, RoverEISGPRTrainer,
    RoverEISGPRTurboSlidingWindowTrainer, RoverEISGPRTurboTrainer,
    RoverLogEISGPRTrainer, RoverLogEISGPRTurboSlidingWindowTrainer,
    RoverLogEISGPRTurboTrainer, ShopLogEISGPRTrainer,
    SigaLogEISGPRTrainer, ValtLogEISGPRTrainer, ZaleLogEISGPRTrainer)
from trainers.svgp_trainer import (
    AdipLogEISVGPTrainer, AdipLogEISVGPTurboSlidingWindowTrainer,
    AdipLogEISVGPTurboTrainer, DhopLogEISVGPTrainer, FexoLogEISVGPTrainer,
    FexoLogEISVGPTurboSlidingWindowTrainer, FexoLogEISVGPTurboTrainer,
    HartmannEISVGPEULBOTrainer, HartmannEISVGPRetrainTrainer,
    HartmannEISVGPTrainer, HartmannEISVGPTurboSlidingWindowTrainer,
    HartmannEISVGPTurboTrainer, LassoDNALogEISVGPTrainer,
    LassoDNALogEISVGPTurboSlidingWindowTrainer,
    LassoDNALogEISVGPTurboTrainer, LunarEISVGPEULBOTrainer,
    LunarEISVGPTrainer, LunarEISVGPTurboSlidingWindowTrainer,
    LunarEISVGPTurboTrainer, Med1LogEISVGPTrainer,
    Med1LogEISVGPTurboSlidingWindowTrainer, Med1LogEISVGPTurboTrainer,
    Med2LogEISVGPTrainer, Med2LogEISVGPTurboSlidingWindowTrainer,
    Med2LogEISVGPTurboTrainer, OsmbLogEISVGPTrainer,
    OsmbLogEISVGPTurboSlidingWindowTrainer, OsmbLogEISVGPTurboTrainer,
    PdopLogEISVGPTrainer, PdopLogEISVGPTurboSlidingWindowTrainer,
    PdopLogEISVGPTurboTrainer, RanoLogEISVGPTrainer,
    RanoLogEISVGPTurboSlidingWindowTrainer, RanoLogEISVGPTurboTrainer,
    RoverEISVGPEULBOTrainer, RoverEISVGPTrainer,
    RoverEISVGPTurboSlidingWindowTrainer, RoverEISVGPTurboTrainer,
    RoverLogEISVGPTrainer, RoverLogEISVGPTurboSlidingWindowTrainer,
    RoverLogEISVGPTurboTrainer, ShopLogEISVGPTrainer,
    SigaLogEISVGPTrainer, ValtLogEISVGPTrainer, ZaleLogEISVGPTrainer)

import botorch
import wandb

arg_trainer_map = {
    'hartmann_ei_exact_gp': HartmannEIExactGPTrainer,
    'hartmann_ei_exact_gp_turbo': HartmannEIExactGPTurboTrainer,
    'hartmann_ei_exact_gp_turbo_sliding_window':
    HartmannEIExactGPTurboSlidingWindowTrainer,
    'hartmann_ei_svgp': HartmannEISVGPTrainer,
    'hartmann_ei_svgp_turbo': HartmannEISVGPTurboTrainer,
    'hartmann_ei_svgp_turbo_sliding_window':
    HartmannEISVGPTurboSlidingWindowTrainer,
    'hartmann_ei_sgpr': HartmannEISGPRTrainer,
    'hartmann_ei_sgpr_turbo': HartmannEISGPRTurboTrainer,
    'hartmann_ei_sgpr_turbo_sliding_window':
    HartmannEISGPRTurboSlidingWindowTrainer,
    'hartmann_ei_svgp_eulbo': HartmannEISVGPEULBOTrainer,
    'hartmann_ei_svgp_retrain': HartmannEISVGPRetrainTrainer,
    'hartmann_ei_ca_gp': HartmannEICaGPTrainer,
    'hartmann_ei_ca_gp_turbo': HartmannEICaGPTurboTrainer,
    'hartmann_ei_ca_gp_turbo_sliding_window':
    HartmannEICaGPTurboSlidingWindowTrainer,
    'hartmann_ei_ca_gp_eulbo': HartmannEICaGPEULBOTrainer,
    'hartmann_log_ei_ca_gp': HartmannLogEICaGPTrainer,
    'lunar_ei_exact_gp': LunarEIExactGPTrainer,
    'lunar_ei_exact_gp_turbo': LunarEIExactGPTurboTrainer,
    'lunar_ei_exact_gp_turbo_sliding_window':
    LunarEIExactGPTurboSlidingWindowTrainer,
    'lunar_ei_ca_gp': LunarEICaGPTrainer,
    'lunar_ei_ca_gp_turbo': LunarEICaGPTurboTrainer,
    'lunar_ei_ca_gp_turbo_sliding_window':
    LunarEICaGPTurboSlidingWindowTrainer,
    'lunar_log_ei_ca_gp': LunarLogEICaGPTrainer,
    'lunar_ei_svgp': LunarEISVGPTrainer,
    'lunar_ei_svgp_turbo': LunarEISVGPTurboTrainer,
    'lunar_ei_svgp_turbo_sliding_window':
    LunarEISVGPTurboSlidingWindowTrainer,
    'lunar_ei_sgpr': LunarEISGPRTrainer,
    'lunar_ei_sgpr_turbo': LunarEISGPRTurboTrainer,
    'lunar_ei_sgpr_turbo_sliding_window':
    LunarEISGPRTurboSlidingWindowTrainer,
    'lunar_ei_svgp_eulbo': LunarEISVGPEULBOTrainer,
    'lunar_ei_ca_gp_eulbo': LunarEICaGPEULBOTrainer,
    'lunar_log_ei_ca_gp_eulbo': LunarLogEICaGPEULBOTrainer,
    'rover_ei_exact_gp': RoverEIExactGPTrainer,
    'rover_ei_exact_gp_turbo': RoverEIExactGPTurboTrainer,
    'rover_ei_exact_gp_turbo_sliding_window':
    RoverEIExactGPTurboSlidingWindowTrainer,
    'rover_log_ei_exact_gp': RoverLogEIExactGPTrainer,
    'rover_log_ei_exact_gp_turbo': RoverLogEIExactGPTurboTrainer,
    'rover_log_ei_exact_gp_turbo_sliding_window':
    RoverLogEIExactGPTurboSlidingWindowTrainer,
    'rover_ei_exact_gp_sliding_window': RoverEIExactGPSlidingWindowTrainer,
    'rover_log_ei_exact_gp_sliding_window':
    RoverLogEIExactGPSlidingWindowTrainer,
    'rover_ei_ca_gp': RoverEICaGPTrainer,
    'rover_ei_ca_gp_turbo': RoverEICaGPTurboTrainer,
    'rover_ei_ca_gp_turbo_sliding_window':
    RoverEICaGPTurboSlidingWindowTrainer,
    'rover_ei_ca_gp_eulbo': RoverEICaGPEULBOTrainer,
    'rover_ei_ca_gp_sliding_window': RoverEICaGPSlidingWindowTrainer,
    'rover_log_ei_ca_gp_sliding_window': RoverLogEICaGPSlidingWindowTrainer,
    'rover_log_ei_ca_gp_turbo': RoverLogEICaGPTurboTrainer,
    'rover_log_ei_ca_gp_turbo_sliding_window':
    RoverLogEICaGPTurboSlidingWindowTrainer,
    'rover_ei_svgp': RoverEISVGPTrainer,
    'rover_ei_svgp_turbo': RoverEISVGPTurboTrainer,
    'rover_ei_svgp_turbo_sliding_window':
    RoverEISVGPTurboSlidingWindowTrainer,
    'rover_ei_sgpr': RoverEISGPRTrainer,
    'rover_ei_sgpr_turbo': RoverEISGPRTurboTrainer,
    'rover_ei_sgpr_turbo_sliding_window':
    RoverEISGPRTurboSlidingWindowTrainer,
    'rover_log_ei_svgp': RoverLogEISVGPTrainer,
    'rover_log_ei_svgp_turbo': RoverLogEISVGPTurboTrainer,
    'rover_log_ei_svgp_turbo_sliding_window':
    RoverLogEISVGPTurboSlidingWindowTrainer,
    'rover_log_ei_sgpr': RoverLogEISGPRTrainer,
    'rover_log_ei_sgpr_turbo': RoverLogEISGPRTurboTrainer,
    'rover_log_ei_sgpr_turbo_sliding_window':
    RoverLogEISGPRTurboSlidingWindowTrainer,
    'rover_ei_svgp_eulbo': RoverEISVGPEULBOTrainer,
    'lasso_dna_log_ei_exact_gp': LassoDNALogEIExactGPTrainer,
    'lasso_dna_log_ei_exact_gp_turbo': LassoDNALogEIExactGPTurboTrainer,
    'lasso_dna_log_ei_exact_gp_turbo_sliding_window':
    LassoDNALogEIExactGPTurboSlidingWindowTrainer,
    'lasso_dna_log_ei_exact_gp_sliding_window':
    LassoDNALogEIExactGPSlidingWindowTrainer,
    'lasso_dna_log_ei_ca_gp': LassoDNALogEICaGPTrainer,
    'lasso_dna_log_ei_ca_gp_turbo': LassoDNALogEICaGPTurboTrainer,
    'lasso_dna_log_ei_ca_gp_turbo_sliding_window':
    LassoDNALogEICaGPTurboSlidingWindowTrainer,
    'lasso_dna_log_ei_ca_gp_sliding_window':
    LassoDNALogEICaGPSlidingWindowTrainer,
    'lasso_dna_log_ei_svgp': LassoDNALogEISVGPTrainer,
    'lasso_dna_log_ei_svgp_turbo': LassoDNALogEISVGPTurboTrainer,
    'lasso_dna_log_ei_svgp_turbo_sliding_window':
    LassoDNALogEISVGPTurboSlidingWindowTrainer,
    'lasso_dna_log_ei_sgpr': LassoDNALogEISGPRTrainer,
    'lasso_dna_log_ei_sgpr_turbo': LassoDNALogEISGPRTurboTrainer,
    'lasso_dna_log_ei_sgpr_turbo_sliding_window':
    LassoDNALogEISGPRTurboSlidingWindowTrainer,
    'osmb_log_ei_exact_gp': OsmbLogEIExactGPTrainer,
    'osmb_log_ei_exact_gp_turbo': OsmbLogEIExactGPTurboTrainer,
    'osmb_log_ei_exact_gp_turbo_sliding_window':
    OsmbLogEIExactGPTurboSlidingWindowTrainer,
    'osmb_log_ei_exact_gp_sliding_window':
    OsmbLogEIExactGPSlidingWindowTrainer,
    'osmb_log_ei_ca_gp': OsmbLogEICaGPTrainer,
    'osmb_log_ei_ca_gp_turbo': OsmbLogEICaGPTurboTrainer,
    'osmb_log_ei_ca_gp_turbo_sliding_window':
    OsmbLogEICaGPTurboSlidingWindowTrainer,
    'osmb_log_ei_ca_gp_sliding_window': OsmbLogEICaGPSlidingWindowTrainer,
    'osmb_log_ei_svgp': OsmbLogEISVGPTrainer,
    'osmb_log_ei_svgp_turbo': OsmbLogEISVGPTurboTrainer,
    'osmb_log_ei_svgp_turbo_sliding_window':
    OsmbLogEISVGPTurboSlidingWindowTrainer,
    'osmb_log_ei_sgpr': OsmbLogEISGPRTrainer,
    'osmb_log_ei_sgpr_turbo': OsmbLogEISGPRTurboTrainer,
    'osmb_log_ei_sgpr_turbo_sliding_window':
    OsmbLogEISGPRTurboSlidingWindowTrainer,
    'fexo_log_ei_exact_gp': FexoLogEIExactGPTrainer,
    'fexo_log_ei_exact_gp_turbo': FexoLogEIExactGPTurboTrainer,
    'fexo_log_ei_exact_gp_turbo_sliding_window':
    FexoLogEIExactGPTurboSlidingWindowTrainer,
    'fexo_log_ei_exact_gp_sliding_window':
    FexoLogEIExactGPSlidingWindowTrainer,
    'fexo_log_ei_gpytorch_exact_gp_sliding_window':
    FexoLogEIGPyTorchExactGPSlidingWindowTrainer,
    'fexo_log_ei_ca_gp_turbo': FexoLogEICaGPTurboTrainer,
    'fexo_log_ei_ca_gp_turbo_sliding_window':
    FexoLogEICaGPTurboSlidingWindowTrainer,
    'fexo_log_ei_ca_gp_sliding_window': FexoLogEICaGPSlidingWindowTrainer,
    'fexo_log_ei_svgp': FexoLogEISVGPTrainer,
    'fexo_log_ei_svgp_turbo': FexoLogEISVGPTurboTrainer,
    'fexo_log_ei_svgp_turbo_sliding_window':
    FexoLogEISVGPTurboSlidingWindowTrainer,
    'fexo_log_ei_sgpr': FexoLogEISGPRTrainer,
    'fexo_log_ei_sgpr_turbo': FexoLogEISGPRTurboTrainer,
    'fexo_log_ei_sgpr_turbo_sliding_window':
    FexoLogEISGPRTurboSlidingWindowTrainer,
    'med1_log_ei_exact_gp': Med1LogEIExactGPTrainer,
    'med1_log_ei_exact_gp_turbo': Med1LogEIExactGPTurboTrainer,
    'med1_log_ei_exact_gp_turbo_sliding_window':
    Med1LogEIExactGPTurboSlidingWindowTrainer,
    'med1_log_ei_exact_gp_sliding_window':
    Med1LogEIExactGPSlidingWindowTrainer,
    'med1_log_ei_ca_gp_turbo': Med1LogEICaGPTurboTrainer,
    'med1_log_ei_ca_gp_turbo_sliding_window':
    Med1LogEICaGPTurboSlidingWindowTrainer,
    'med1_log_ei_ca_gp_sliding_window': Med1LogEICaGPSlidingWindowTrainer,
    'med1_log_ei_svgp': Med1LogEISVGPTrainer,
    'med1_log_ei_svgp_turbo': Med1LogEISVGPTurboTrainer,
    'med1_log_ei_svgp_turbo_sliding_window':
    Med1LogEISVGPTurboSlidingWindowTrainer,
    'med1_log_ei_sgpr': Med1LogEISGPRTrainer,
    'med1_log_ei_sgpr_turbo': Med1LogEISGPRTurboTrainer,
    'med1_log_ei_sgpr_turbo_sliding_window':
    Med1LogEISGPRTurboSlidingWindowTrainer,
    'med2_log_ei_exact_gp': Med2LogEIExactGPTrainer,
    'med2_log_ei_exact_gp_turbo': Med2LogEIExactGPTurboTrainer,
    'med2_log_ei_exact_gp_turbo_sliding_window':
    Med2LogEIExactGPTurboSlidingWindowTrainer,
    'med2_log_ei_exact_gp_sliding_window':
    Med2LogEIExactGPSlidingWindowTrainer,
    'med2_log_ei_ca_gp_turbo': Med2LogEICaGPTurboTrainer,
    'med2_log_ei_ca_gp_turbo_sliding_window':
    Med2LogEICaGPTurboSlidingWindowTrainer,
    'med2_log_ei_ca_gp_sliding_window': Med2LogEICaGPSlidingWindowTrainer,
    'med2_log_ei_svgp': Med2LogEISVGPTrainer,
    'med2_log_ei_svgp_turbo': Med2LogEISVGPTurboTrainer,
    'med2_log_ei_svgp_turbo_sliding_window':
    Med2LogEISVGPTurboSlidingWindowTrainer,
    'med2_log_ei_sgpr': Med2LogEISGPRTrainer,
    'med2_log_ei_sgpr_turbo': Med2LogEISGPRTurboTrainer,
    'med2_log_ei_sgpr_turbo_sliding_window':
    Med2LogEISGPRTurboSlidingWindowTrainer,
    'pdop_log_ei_exact_gp_sliding_window':
    PdopLogEIExactGPSlidingWindowTrainer,
    'pdop_log_ei_exact_gp': PdopLogEIExactGPTrainer,
    'pdop_log_ei_exact_gp_turbo': PdopLogEIExactGPTurboTrainer,
    'pdop_log_ei_exact_gp_turbo_sliding_window':
    PdopLogEIExactGPTurboSlidingWindowTrainer,
    'pdop_log_ei_ca_gp_turbo': PdopLogEICaGPTurboTrainer,
    'pdop_log_ei_ca_gp_turbo_sliding_window':
    PdopLogEICaGPTurboSlidingWindowTrainer,
    'pdop_log_ei_ca_gp_sliding_window': PdopLogEICaGPSlidingWindowTrainer,
    'pdop_log_ei_svgp': PdopLogEISVGPTrainer,
    'pdop_log_ei_svgp_turbo': PdopLogEISVGPTurboTrainer,
    'pdop_log_ei_svgp_turbo_sliding_window':
    PdopLogEISVGPTurboSlidingWindowTrainer,
    'pdop_log_ei_sgpr': PdopLogEISGPRTrainer,
    'pdop_log_ei_sgpr_turbo': PdopLogEISGPRTurboTrainer,
    'pdop_log_ei_sgpr_turbo_sliding_window':
    PdopLogEISGPRTurboSlidingWindowTrainer,
    'adip_log_ei_exact_gp': AdipLogEIExactGPTrainer,
    'adip_log_ei_exact_gp_turbo': AdipLogEIExactGPTurboTrainer,
    'adip_log_ei_exact_gp_turbo_sliding_window':
    AdipLogEIExactGPTurboSlidingWindowTrainer,
    'adip_log_ei_exact_gp_sliding_window':
    AdipLogEIExactGPSlidingWindowTrainer,
    'adip_log_ei_ca_gp_turbo': AdipLogEICaGPTurboTrainer,
    'adip_log_ei_ca_gp_turbo_sliding_window':
    AdipLogEICaGPTurboSlidingWindowTrainer,
    'adip_log_ei_ca_gp_sliding_window': AdipLogEICaGPSlidingWindowTrainer,
    'adip_log_ei_sgpr': AdipLogEISGPRTrainer,
    'adip_log_ei_sgpr_turbo': AdipLogEISGPRTurboTrainer,
    'adip_log_ei_sgpr_turbo_sliding_window':
    AdipLogEISGPRTurboSlidingWindowTrainer,
    'adip_log_ei_svgp': AdipLogEISVGPTrainer,
    'adip_log_ei_svgp_turbo': AdipLogEISVGPTurboTrainer,
    'adip_log_ei_svgp_turbo_sliding_window':
    AdipLogEISVGPTurboSlidingWindowTrainer,
    'rano_log_ei_exact_gp': RanoLogEIExactGPTrainer,
    'rano_log_ei_exact_gp_turbo': RanoLogEIExactGPTurboTrainer,
    'rano_log_ei_exact_gp_turbo_sliding_window':
    RanoLogEIExactGPTurboSlidingWindowTrainer,
    'rano_log_ei_exact_gp_sliding_window':
    RanoLogEIExactGPSlidingWindowTrainer,
    'rano_log_ei_ca_gp_turbo': RanoLogEICaGPTurboTrainer,
    'rano_log_ei_ca_gp_turbo_sliding_window':
    RanoLogEICaGPTurboSlidingWindowTrainer,
    'rano_log_ei_ca_gp_sliding_window': RanoLogEICaGPSlidingWindowTrainer,
    'rano_log_ei_svgp': RanoLogEISVGPTrainer,
    'rano_log_ei_svgp_turbo': RanoLogEISVGPTurboTrainer,
    'rano_log_ei_svgp_turbo_sliding_window':
    RanoLogEISVGPTurboSlidingWindowTrainer,
    'rano_log_ei_sgpr': RanoLogEISGPRTrainer,
    'rano_log_ei_sgpr_turbo': RanoLogEISGPRTurboTrainer,
    'rano_log_ei_sgpr_turbo_sliding_window':
    RanoLogEISGPRTurboSlidingWindowTrainer,
    'siga_log_ei_ca_gp_sliding_window': SigaLogEICaGPSlidingWindowTrainer,
    'siga_log_ei_svgp': SigaLogEISVGPTrainer,
    'siga_log_ei_sgpr': SigaLogEISGPRTrainer,
    'zale_log_ei_ca_gp_sliding_window': ZaleLogEICaGPSlidingWindowTrainer,
    'zale_log_ei_svgp': ZaleLogEISVGPTrainer,
    'zale_log_ei_sgpr': ZaleLogEISGPRTrainer,
    'valt_log_ei_ca_gp_sliding_window': ValtLogEICaGPSlidingWindowTrainer,
    'valt_log_ei_svgp': ValtLogEISVGPTrainer,
    'valt_log_ei_sgpr': ValtLogEISGPRTrainer,
    'dhop_log_ei_ca_gp_sliding_window': DhopLogEICaGPSlidingWindowTrainer,
    'dhop_log_ei_svgp': DhopLogEISVGPTrainer,
    'dhop_log_ei_sgpr': DhopLogEISGPRTrainer,
    'shop_log_ei_ca_gp_sliding_window': ShopLogEICaGPSlidingWindowTrainer,
    'shop_log_ei_svgp': ShopLogEISVGPTrainer,
    'shop_log_ei_sgpr': ShopLogEISGPRTrainer,
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
    parser.add_argument('--spherical_linear_lengthscale_prior',
                        default='dsp_unscaled',
                        help='lengthscale prior for spherical linear kernel')
    parser.add_argument(
        '--use_ard_kernel',
        action='store_true',
        help='fit a separate lengthscale for each input dimension')
    parser.add_argument('--use_greedy_decoding',
                        action='store_true',
                        help='greedily decode from VAE for mol design tasks')
    parser.add_argument(
        '--turn_on_outcome_transform',
        action='store_true',
        help='turn on standardize outcome transform for non exact GPs')
    parser.add_argument('--turn_on_input_transform',
                        action='store_true',
                        help='turn on normalize transform for non exact GPs')
    parser.add_argument('--turn_on_botorch_input_transform',
                        action='store_true',
                        help='turn on normalize transform for inputs of GPs')
    parser.add_argument(
        '--turn_on_simple_input_transform',
        action='store_true',
        help='turn on normalize transform for inputs of GPs to unit cube')
    parser.add_argument(
        '--turn_on_sobol_init',
        action='store_true',
        help='use scrambled sobol sequences for BO initialization')
    parser.add_argument(
        '--use_faithful_turbo_restart',
        action='store_true',
        help=
        'for TurBO exact GPs, restart with a fresh local initial design instead of only resetting the trust-region state'
    )
    parser.add_argument('--use_output_scale',
                        action='store_true',
                        help='use outputscale with spherical linear kernel')
    parser.add_argument('--remove_global_ls',
                        action='store_true',
                        help='turn off global ls in spherical linear kernel')
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
    parser.add_argument('--reinit_model_complete',
                        action='store_true',
                        help='reinitialize entire CaGP model at each step')
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
    parser.add_argument('--turn_off_prior',
                        action='store_true',
                        help='turn off priors in spherical linear settings')
    parser.add_argument('--add_likelihood_to_posterior',
                        action='store_true',
                        help='add likelihood to posterior for EI')
    parser.add_argument('--notes', default='', help='note on experiment run')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--use_analytic_acq_func',
                        action='store_true',
                        help='use analytic acquisition function instead of MC')
    parser.add_argument('--enable_raasp',
                        action='store_true',
                        help='enable RAASP sampling in AF optimization')
    parser.add_argument('--use_parallel_mode',
                        action='store_true',
                        help='enable custom batched lbfgs in AF optimization')
    parser.add_argument('--raasp_best_pct',
                        default=5,
                        type=float,
                        help='pct of best points to perturb in RAASP')
    parser.add_argument('--raasp_sigma',
                        default=1e-3,
                        type=float,
                        help='std of perturbations in RAASP')
    parser.add_argument('--ln_noise_prior_loc',
                        default=-4.0,
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

    # set botorch logging to debug
    botorch.settings.log_level(logging.DEBUG)

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
