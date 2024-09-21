"""
PANDA hypersweep args:

python3 hypersweep_optuna.py  
--project_root_dir /mnt/d/github/slideflow/panda_project 
--annotations //mnt/d/github/slideflow/datasets/panda/pandas_tumor_subtyping_clean985.csv 
--tile-px 244 
--tile-um 112 
--experiment_name hyperout_attention_mil
--splits_config /mnt/d/github/slideflow/notebooks/split_config.json
--outdir_model panda_project/mlflow_test02/

"""

import argparse
import gc
import json
import os
import time
import torch

import mlflow
import mlflow.pytorch
import optuna
from optuna.samplers import TPESampler
# import slideflow as sf

from auto_train_features import *


def objective(trial, args):

    #store the original output dir
    original_outdir_model = args.outdir_model

    if args.outdir_model:
        args.outdir_model = os.path.join(original_outdir_model, str(trial.number))

    args.name = str(trial.number)

    #Hyperparameters to be optimized
    args.lr = trial.suggest_float("lr", 0.0001, 0.010, step=0.001)
    if args.model in ['clam_sb', 'clam_mb', 'mil_fc_mc']:
        args.weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.0005, step=0.0001)
        # args.optimizer =  trial.suggest_categorical("optimizer",['adam', "adamw","sgd"])

    args.bag_size = trial.suggest_int("bag_size", 500, 3000, step= 500)
    args.extractor_model = trial.suggest_categorical("extractor_model", ['ctranspath','histossl','plip',
                                                                        'retccl','resnet50_imagenet',
                                                                        'alexnet_imagenet', 'inception_imagenet', 
                                                                        'resnet18_imagenet', 'resnext50_32x4d_imagenet', 
                                                                        'wide_resnet50_2_imagenet'])

    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)      

    metrics_dict = main(args, trial)
    optimize_metric = metrics_dict['accuracy']

    #Reset the output dir to the original
    args.outdir_model = original_outdir_model
    torch.cuda.empty_cache()
    gc.collect()

    return optimize_metric



if __name__=="__main__":

    args = parse_args()
    study = optuna.create_study(
        study_name=args.experiment_name,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(n_startup_trials=15),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=30, n_warmup_steps=5, interval_steps=10
        ),
    )
    study.optimize(lambda trial: objective(trial, args), n_trials=80, catch=(RuntimeError,))