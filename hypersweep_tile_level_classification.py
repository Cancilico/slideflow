"""
IDH1 hypersweep args:

python3 hypersweep_optuna.py  
--project_root_dir /mnt/d/projects/slideflow/idh1_project 
--annotations /mnt/d/projects/slideflow/idh1_project/idh1_tumor_vs_normal_trainVal_130_slides.csv
--slides_dir /mnt/d/DATA/idh1_wsi
--dataset_config /mnt/d/projects/slideflow/datasets/idh1/idh1_tumor.json
--project_name idh1_project
--load_project
--load_dataset
--train_classifier
--val_strategy fixed
--val_fraction 0.2
--epochs 100
--max_tiles 500
--tile-px 244 
--tile-um 112 
--experiment_name hyperout_attention_mil
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

from auto_train_tile_level import *


def objective(trial, args):

    #store the original output dir
    original_outdir_model = args.outdir_model

    if args.outdir_model:
        args.outdir_model = os.path.join(original_outdir_model, str(trial.number))

    args.name = str(trial.number)
    
    args.model = trial.suggest_categorical("model",['resnet18','resnet50', 'alexnet','squeezenet','densenet', 
                'googlenet','resnext50_32x4d','mobilenet_v2', 'mobilenet_v3_small', 
                'mobilenet_v3_large','wide_resnet50_2','mnasnet','xception','nasnet_large'])
    # Hypothesize that the number of hidden layers and the width of each layer could be important
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)  # e.g., between 1 and 5 hidden layers
    hidden_layer_width = trial.suggest_int("hidden_layer_width", 50, 500)  # e.g., each layer has between 50 and 500 neurons

    # Setup args to reflect the chosen structure
    args.hidden_layers = num_hidden_layers  # Integer: the number of hidden layers
    args.hidden_layer_width = hidden_layer_width 
        
    #Hyperparameters to be optimized
    args.optimizer =  trial.suggest_categorical("optimizer",['Adadelta','Adagrad','Adam','AdamW','SparseAdam',
                                                              'Adamax','ASGD','LBFGS','RMSprop','Rprop', 'SGD'])
    args.lr = trial.suggest_float("lr", 0.0001, 0.010, step=0.001)
    args.learning_rate_decay = trial.suggest_float("learning_rate_decay", 0.0001, 0.0005, step=0.0001)
    args.max_tiles = trial.suggest_int("max_tiles", 500, 10000, step= 500)
    args.learning_rate_decay_steps = trial.suggest_int("learning_rate_decay_steps", 1000, 10000, step=500) 
    args.early_stop_patience = 10

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