"""
python3 auto_train_tile_level.py
--project_root_dir /mnt/d/projects/slideflow/idh1_project 
--annotations /mnt/d/projects/slideflow/idh1_project/idh1_tumor_vs_normal_trainVal_130_slides.csv
--slides_dir /mnt/d/DATA/idh1_wsi
--dataset_config /mnt/d/github/slideflow/datasets/panda/panda_subtypes.json 
--project_name panda_project 
--tile-px 224 
--epochs 2 
--quality-control otsu 
--val_strategy fixed 
--val_fraction 0.2 
--model xception
--tile-um 112 
--outdir_model panda_project/tile_classification/ 
--label_column label 
--load_project 
--load_dataset
--train_classifier
--batch_size 
--max_tiles 100
--experiment_name tile_mlflow_test

"""

import time
import json
import argparse
import optuna

import slideflow as sf
from slideflow.slide import qc
from slideflow.mil import mil_config
from slideflow.mil import train_mil

import optuna
import mlflow
mlflow.set_tracking_uri("file:////mnt/d/projects/slideflow/idh1_project/mlruns_tile_test")
# mlflow.set_tracking_uri("https://mlflow.cancili.co/")



def parse_args():
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--project_root_dir', type=str, default=None, 
                        help='project directory')
    parser.add_argument('--annotations', type=str, default=None, 
                        help='Path to label csv file')
    parser.add_argument('--slides_dir', type=str, default='/mnt/d/DATA/idh1_wsi', 
                        help='Path to slides')
    parser.add_argument('--dataset_config', type=str, default='/mnt/d/projects/slideflow/datasets/idh1/idh1_tumor.json', 
                        help='Path to data config file')
    parser.add_argument('--project_name', type=str, default='idh1_project', 
                        help='Set the project name.')
    parser.add_argument('--load_project', default=False, action="store_true", 
                        help='Load project')
    parser.add_argument('--load_dataset', default=False, action="store_true", 
                        help='Load dataset')
    parser.add_argument('--train_classifier', default=False, action="store_true", 
                        help='To train classfier model')
    parser.add_argument('--tile-px', default = 2560, type=int,
                        help='Size of tiles to extract, in pixels.')
    parser.add_argument('--tile-um', default = "174", type = str,
                        help='Size of tiles to extract, in microns')
    parser.add_argument("-qc", "--quality-control", default='both',
                        help='Apply quality control to tiles, removing blurry or background tiles.  Options: otsu, blur, both')
    # parser.add_argument('--tfrecords_dir', type=str, default=None, 
    #                     help='Path to dataset slide tfrecords')
    parser.add_argument('--label_column', default = "label", type = str,
                        help='column name containg slide labels in teh annotation csv file')  
    parser.add_argument('--val_strategy', default = "fixed", type = str,
                        help='Select a validation stratergy: k-fold, k-fold-preserved-site, bootstrap, or fixed')    
    parser.add_argument('--val_fraction', default = 0.2, type = float,
                        help='Fraction of dataset to use for validation testing, if strategy is "fixed".')
    parser.add_argument('--val_k_fold', default = None, type = float,
                        help='Total number of K if using K-fold validation. Defaults to 3.') 
    parser.add_argument('--model', type=str, default='resnet50', 
                        help='Select MIL model: xceptionnet, resnet18, resnet50, ')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to cp.ckpt from which to load weights. Defaults to None.')
    parser.add_argument('--lr', default = 1e-4, type = int,
                        help='learning rate (default: 0.0001)')
    parser.add_argument("--learning_rate_decay", type=float, default=None, help="Weight decay for regularization")
    parser.add_argument('--learning_rate_decay_steps', type=int, default=10000, help='Number of steps between learning rate decay')
    parser.add_argument('--batch_size', default = 32, type = int,
                        help='Set batch size for training mil model')
    parser.add_argument('--epochs', default = 50, type = int,
                        help='number of epochs to train')
    parser.add_argument('--min_tiles', default = 500, type = int,
                        help='Minimum number of tiles a slide must have to include in training. Defaults to 0.')
    parser.add_argument('--max_tiles', default = 1000, type = int,
                        help=' Only use up to this many tiles from each slide for training. Defaults to 0 (include all tiles)')
    parser.add_argument('--mixed_precision', default=True, action="store_true", 
                        help='Enable mixed precision. Defaults to True.')
    parser.add_argument('--multi_gpu', default=False, action="store_true", 
                        help='Train using multiple GPUs when available. Defaults to False.')
    parser.add_argument('--outdir_model', type=str, default=None, 
                        help='Path to save the mil model and predictions') 
    parser.add_argument('--experiment_name', type=str, default=None, 
                        help='Path to save the mil model and predictions') 
    parser.add_argument("--name", default="default", help="name of the training run")
    parser.add_argument("--optimizer", default="adam", help="name of the optimizer")     
    parser.add_argument("--hidden_layers", default=10, type= int, help="Add final layers to models") 
    parser.add_argument('--hidden_layer_width', type=int, default=10, help='Width of each hidden layer')
    parser.add_argument("--early_stop_patience", default=10,type= int,  help="Patience for early stopping") 
    parser.add_argument("--early_stop_method", default='accuracy',help="Select a method to peform early stopping: manual, accuracy, loss")  
    
    args = parser.parse_args()

    # tile_um can be int or str, but always comes as string from the command line. Convert digit-string to ints
    if args.tile_um is not None and args.tile_um.isdigit():
        args.tile_um = int(args.tile_um)

    return args


def train_classifier(args, 
                    dataset, 
                    P
                ):
    hp = sf.ModelParams(
        tile_px = args.tile_px, 
        tile_um = args.tile_um,
        model = args.model,
        batch_size = args.batch_size,
        epochs = [args.epochs],
        learning_rate = args.lr,
        learning_rate_decay = args.learning_rate_decay,
        learning_rate_decay_steps = args.learning_rate_decay_steps,
        optimizer = args.optimizer,
        hidden_layers = args.hidden_layers,
        early_stop_patience = args.early_stop_patience,
        early_stop_method = args.early_stop_method
    )    
    start_train = time.time()
    mlflow.start_run(run_name=args.name)
    metric_dict =P.train(outcomes = args.label_column,
                                params=hp,
                                dataset =  dataset, #train_dataset,
                                # val_dataset=val_dataset,
                                # model_type = 'classification',
                                filters= None,
                                filter_blank=None, 
                                min_tiles = args.min_tiles,
                                max_tiles = args.max_tiles,
                                val_strategy= args.val_strategy,
                                val_fraction = args.val_fraction,
                                mixed_precision = args.mixed_precision,
                                multi_gpu = args.multi_gpu,
                                checkpoint = args.checkpoint,
                                mlflow=mlflow,
                                )    
    end_train = time.time()
    time_training = end_train - start_train
    print('time_training:', time_training)
    # return train_acc, val_loss, val_acc
    return metric_dict
   


def main(args, trial=None):

    # Load Project
    P = sf.load_project(args.project_root_dir) 
    # Load extracted tile-datset
    dataset = P.dataset(tile_px=args.tile_px, 
                        tile_um=args.tile_um,
                        )
    metric_dict = train_classifier(args,dataset, P)

    if trial.should_prune():
        raise optuna.TrialPruned()
    mlflow.end_run()

    return metric_dict


if __name__ == "__main__":
    start_time = time.time()

    args = parse_args()
    print("Parsed Command-Line Arguments:")
    for arg in vars(args):
        value = getattr(args, arg)
        print(f"{arg}: {value}")


    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)        

    results = main(args)    
    print('results:', results)
    print("finished run!")

    end_time = time.time()
    total_time = end_time -start_time
    print('total_time: ',total_time)