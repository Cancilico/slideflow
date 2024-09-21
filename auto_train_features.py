"""
PANDA dataset

python3 end2end_mil_train.py 
--project_root_dir /mnt/d/github/slideflow/panda_project 
--annotations //mnt/d/github/slideflow/datasets/panda/pandas_tumor_subtyping_clean985.csv 
--slides_dir /mnt/d/DATA/PANDA/pandaChallege2020/prostate-cancer-grade-assessment_wsi/train_images 
--dataset_config /mnt/d/github/slideflow/datasets/panda/panda_subtypes.json 
--project_name panda_project 
--tile-px 244 
--quality-control otsu 
--extractor_model resnet18_imagenet 
--outdir_bags panda_project/feature_bags 
--load_project 
--load_dataset 
--tile-um 112 
--load_features 
--train_mil 
--val_fraction 0.1 
--val_strategy fixed 
--splits_config /mnt/d/github/slideflow/notebooks/split_config.json 
--outdir_model panda_project/  
--label_column label
--experiment_name 

------------------------------------------------------------------------------------------
IDH1 dataset:

export SF_SLIDE_BACKEND=libvips

python3 end2end_mil_train.py
--project_root_dir idh1_project
--annotations /mnt/d/github/slideflow/datasets/idh1/idh1_tumor_vs_normal_clean165.csv
--slides_dir /mnt/d/github/end2end-WSI-preprocessing/wsi
--dataset_config /mnt/d/github/slideflow/datasets/idh1/idh1_tumor.json
--project_name idh1_project
--tile-px 224
--tile-um 112
--quality-control otsu
--extractor_model ctranspath
--outdir_bags idh1_project/feature_bags
--val_strategy fixed
--val_fraction 0.2
--model clam_sb
--outdir_model idh1_project/model_clam_sb/

"""

import time
import json
import argparse
import optuna

import slideflow as sf
from slideflow.slide import qc
from slideflow.mil import mil_config
from slideflow.mil import train_mil

import mlflow
# mlflow.set_tracking_uri("file:////mnt/d/github/slideflow/mlruns_test")
mlflow.set_tracking_uri("https://mlflow.cancili.co/")


def parse_args():
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--project_root_dir', type=str, default=None, 
                        help='project directory')
    parser.add_argument('--annotations', type=str, default=None, 
                        help='Path to label csv file')
    parser.add_argument('--tile-px', default = 2560, type=int,
                        help='Size of tiles to extract, in pixels.')
    parser.add_argument('--tile-um', default = "112", type = str,
                        help='Size of tiles to extract, in microns')
    parser.add_argument('--extractor_model', default = "ctranspath", type = str,
                        help='Select a feature extractor model to get corresponding feature bags')
    parser.add_argument('--splits_config', type=str, default=None, 
                        help='Path to splits config json file contatainig slide-ids for train and val')
    parser.add_argument('--val_strategy', default = "fixed", type = str,
                        help='Select a validation stratergy: k-fold, k-fold-preserved-site, bootstrap, or fixed')  
    parser.add_argument('--label_column', default = "label", type = str,
                        help='column name containg slide labels in teh annotation csv file')    
    parser.add_argument('--val_fraction', default = 0.2, type = float,
                        help='Fraction of validation split') 
    parser.add_argument('--model', type=str, default='attention_mil', 
                        help='Select MIL model: attention_mil, transmil,, clam_sb, clam_mb, mil_fc_mc')
    parser.add_argument('--lr', default = 1e-4, type = int,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', default = 1e-4, type = int,
                        help='Weight decay. Only used if ``fit_one_cycle=False``. Defaults to 1e-5.')
    parser.add_argument('--batch_size', default = 64, type = int,
                        help='Set batch size for training mil model')
    parser.add_argument('--bag_size', default = 512, type = int,
                        help=' Bag size. Defaults to 512.')
    parser.add_argument('--epochs', default = 50, type = int,
                        help='number of epochs to train')
    parser.add_argument('--outdir_model', type=str, default=None, 
                        help='Path to save the mil model and predictions') 
    parser.add_argument('--experiment_name', type=str, default=None, 
                        help='Path to save the mil model and predictions') 
    parser.add_argument("--name", default="default", help="name of the training run")

    args = parser.parse_args()

    # tile_um can be int or str, but always comes as string from the command line. Convert digit-string to ints
    if args.tile_um is not None and args.tile_um.isdigit():
        args.tile_um = int(args.tile_um)

    return args


def create_splits(dataset, args):
    train, val = dataset.split(labels = args.label_column, 
                               val_strategy = args.val_strategy,
                               val_fraction= args.val_fraction,
                               splits = args.splits_config,                           
                               )
    return train, val


def train_mil_features(args,train, val ):

    config_args = {
        'model': args.model,
        'lr': args.lr,
        'wd': args.weight_decay,
        'batch_size': args.batch_size,
        'bag_size':args.bag_size,
        'epochs': args.epochs,
        'fit_one_cycle': True,
    }
    if args.model in ['clam_sb', 'clam_mb', 'mil_fc_mc']:
        config_args['B'] = 45
    # Configure the MIL model for the current combination
    config = mil_config(**config_args)

    # bags_dir = f"{args.outdir_bags}/bags_{args.extractor_model}"
    bags_dir =  f"{args.project_root_dir}/feature_bags/bags_{args.extractor_model}"
    print('Loading feature bags from... ',bags_dir )
    outdir = f"{args.outdir_model}/model_{args.model}_bags_{args.extractor_model}"
    print('Saving model and result at... ',outdir )    

    start_train = time.time()
    mlflow.start_run(run_name=args.name)
    _, metric_dict = train_mil(
                        config,
                        train_dataset = train,
                        val_dataset = val,
                        outcomes = args.label_column, #'label',
                        bags = bags_dir, #'panda_project/bags/',
                        outdir = outdir, #'panda_project/model_clam_sb/',
                        mlflow=mlflow,
                    )
    end_train = time.time()
    time_training = end_train - start_train
    print('time_training:', time_training)
    return metric_dict


def main(args, trial=None):

    # Load Project
    P = sf.load_project(args.project_root_dir) 
    # Load extracted tile-datset
    dataset = P.dataset(tile_px=args.tile_px, 
                        tile_um=args.tile_um,
                        )
    train, val = create_splits(dataset, args)
    metric_dict = train_mil_features(args,train, val )

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