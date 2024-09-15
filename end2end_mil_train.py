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

import slideflow as sf
from slideflow.mil import mil_config
from slideflow.mil import train_mil

import mlflow
mlflow.set_tracking_uri("file:////mnt/d/github/slideflow/mlruns_test")


parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--project_root_dir', type=str, default=None, 
                    help='project directory')
parser.add_argument('--annotations', type=str, default=None, 
                    help='Path to label csv file')
parser.add_argument('--slides_dir', type=str, default=None, 
                    help='Path to slides')
parser.add_argument('--dataset_config', type=str, default=None, 
                    help='Path to data config file')
parser.add_argument('--project_name', type=str, default=None, 
                    help='Path to data config file')
parser.add_argument('--load_project', default=False, action="store_true", 
                    help='Load project')
parser.add_argument('--load_dataset', default=False, action="store_true", 
                    help='Load dataset')
parser.add_argument('--load_features', default=False, action="store_true", 
                    help='Load features')
parser.add_argument('--train_mil', default=False, action="store_true", 
                    help='To train MIL model')
parser.add_argument('--tile-px', default = 2560, type=int,
                    help='Size of tiles to extract, in pixels.')
parser.add_argument('--tile-um', default = "174", type = str,
                    help='Size of tiles to extract, in microns')
parser.add_argument("-qc", "--quality-control", default='both',
                    help='Apply quality control to tiles, removing blurry or background tiles.  Options: otsu, blur,')
parser.add_argument('--extractor_model', default = "ctranspath", type = str,
                    help='Select a feature extractor model')
parser.add_argument('--outdir_bags', type=str, default=None, 
                    help='Path to feature bags saving dir')
parser.add_argument('--splits_config', type=str, default=None, 
                    help='Path to splits config json file contatainig slide-ids for train and val')
parser.add_argument('--val_strategy', default = "fixed", type = str,
                    help='Select a validation stratergy: k-fold, k-fold-preserved-site, bootstrap, or fixed')  
parser.add_argument('--label_column', default = "label", type = str,
                    help='column name containg slide labels in teh annotation csv file')    
parser.add_argument('--val_fraction', default = None, type = float,
                    help='Fraction of validation split') 
parser.add_argument('--model', type=str, default='attention_mil', 
                    help='Select MIL model: attention_mil, transmil,, clam_sb, clam_mb, mil_fc_mc')
parser.add_argument('--lr', default = 1e-4, type = int,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--batch_size', default = 32, type = int,
                    help='Set batch size for training mil model')
parser.add_argument('--epochs', default = 50, type = int,
                    help='number of epochs to train')
parser.add_argument('--outdir_model', type=str, default=None, 
                    help='Path to save the mil model and predictions') 
parser.add_argument('--experiment_name', type=str, default=None, 
                    help='Path to save the mil model and predictions') 
parser.add_argument("--name", default="default", help="name of the training run") 
         

def create_project(args):

    # Part 01: Create project= panda_project
    start_create_project = time.time()
    P = sf.create_project(
        root= args.project_root_dir,
        annotations = args.annotations,
        slides = args.slides_dir,
        dataset_config = args.dataset_config,
        name = args.project_name,
        )
    end_create_project = time.time()
    time_create_project = end_create_project - start_create_project
    print('time_create_project:', round(time_create_project, 2) )
    return P
    

def extract_tile_dataset(P, args):

    # Part 02: Extract tiles from slides
    # P = sf.load_project('/mnt/d/github/slideflow/panda_project')
    start_extract_tiles = time.time()
    dataset = P.extract_tiles(
                    tile_px =  args.tile_px,  #244,  # Tile size, in pixels
                    tile_um = args.tile_um,   #112 Tile size, in microns
                    qc = args.quality_control, #otsu
                    img_format = 'png',                    
                    )
    end_extract_tiles = time.time()
    time_extract_tiles = end_extract_tiles - start_extract_tiles
    print('time_extract_tiles:', round(time_extract_tiles, 2)) 
    return dataset


def filter_config(model_config, exclude_params):
    """
    Remove keys from model_config that are listed in exclude_params.
    
    :param model_config: Dictionary of the configuration parameters for a model.
    :param exclude_params: List of keys to be excluded from the configuration.
    :return: Filtered configuration dictionary.
    """
    return {k: v for k, v in model_config.items() if k not in exclude_params}

def get_extractor_config(model_name, args):
    """ Return specific configuration based on the model name """
    # Base configuration applied to all models unless explicitly overwritten
    base_config = {
        'backend': 'torch',
        'tile_px': args.tile_px,        
    }
    # Specific configurations for models that need unique settings
    specific_configs = {
        'inception_imagenet': {'resize': 224},  # Different resize for Inception
        'ctranspath': {'resize': 224, 'center_crop': True, 'interpolation': 'bicubic', 'exclude': ['tile_px']},  # Exclude tile_px
    }
    # Retrieve specific configuration and handle exclusions if any
    model_config = specific_configs.get(model_name, {})
    exclude_params = model_config.pop('exclude', [])
    filtered_config = filter_config({**base_config, **model_config}, exclude_params)
    
    return filtered_config


def extract_features(args, dataset):
    start_extract_feats = time.time()
    # Get the specific configuration for the feature extractor
    extractor_config = get_extractor_config(args.extractor_model, args)
    # Build the feature extractor with the specific configuration
    feature_extractor = sf.build_feature_extractor(args.extractor_model, **extractor_config)

    outdir = f'{args.outdir_bags}/bags_{args.extractor_model}'
    dataset.generate_feature_bags(feature_extractor, 
                                  outdir = outdir
                                  )
    end_extract_feats = time.time()
    time_extract_feats = end_extract_feats - start_extract_feats
    print('time_extract_feats:', time_extract_feats) 


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
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'fit_one_cycle': True,
    }

    if args.model in ['clam_sb', 'clam_mb', 'mil_fc_mc']:
        config_args['B'] = 45

    # Configure the MIL model for the current combination
    config = mil_config(**config_args)

    bags_dir = f"{args.outdir_bags}/bags_{args.extractor_model}"
    print('Loading feature bags from... ',bags_dir )
    outdir = f"{args.outdir_model}/model_{args.model}_bags_{args.extractor_model}"
    print('Saving model and result at... ',outdir )    

    # Part 06: Train MIL model
    start_train = time.time()
    mlflow.start_run(run_name=args.name)
    train_mil(
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


def main(args):

    if args.load_project:
        P = sf.load_project(args.project_root_dir)
    else:
        P = create_project(args)

    # Part 02: Extract tiles from slides    
    if args.load_dataset:
        # Load extracted tile-datset
        dataset = P.dataset(tile_px=args.tile_px, tile_um=args.tile_um)
    else:
        dataset = extract_tile_dataset(P, args)
    
    if not args.load_features:
        extract_features(args, dataset)

    if args.train_mil:
        train, val = create_splits(dataset, args)
        train_mil_features(args,train, val )

    return 



if __name__ == "__main__":
    start_time = time.time()

    args = parser.parse_args()
    # tile_um can be int or str, but always comes as string from the command line. Convert digit-string to ints
    if args.tile_um is not None and args.tile_um.isdigit():
        args.tile_um = int(args.tile_um)

    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)        

    results = main(args)
    print("finished run!")

    end_time = time.time()
    total_time = end_time -start_time
    print('total_time: ',total_time)