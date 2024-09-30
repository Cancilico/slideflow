"""
python3 tile_level_classification.py
--project_root_dir panda_project 
--annotations /mnt/d/github/slideflow/panda_project/pandas_tumor_subtyping_train_val_clean807_dataset.csv
--slides_dir /mnt/d/DATA/PANDA/pandaChallege2020/prostate-cancer-grade-assessment_wsi/train_images 
--dataset_config /mnt/d/github/slideflow/datasets/panda/panda_subtypes.json 
--project_name panda_project 
--tile-px 224 
--epochs 2 
--quality-control otsu 
--val_strategy fixed 
--val_fraction 0.2 
--model xception
--tile-um 112 
--outdir_model panda_project/classification/ 
--splits_config /mnt/d/github/slideflow/notebooks/split_config.json 
--label_column label 
--load_project 
--load_dataset
--train_classifier
--batch_size 

"""

import time
import json
import argparse

import slideflow as sf

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
parser.add_argument('--val_fraction', default = None, type = float,
                    help='Fraction of dataset to use for validation testing, if strategy is "fixed".')
parser.add_argument('--val_k_fold', default = None, type = float,
                    help='Total number of K if using K-fold validation. Defaults to 3.') 
parser.add_argument('--model', type=str, default='resnet50', 
                    help='Select MIL model: xceptionnet, resnet18, resnet50, ')
parser.add_argument('--checkpoint', type=str, default=None, 
                    help='Path to cp.ckpt from which to load weights. Defaults to None.')
parser.add_argument('--lr', default = 1e-4, type = int,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--batch_size', default = 32, type = int,
                    help='Set batch size for training mil model')
parser.add_argument('--epochs', default = 50, type = int,
                    help='number of epochs to train')
parser.add_argument('--min_tiles', default = 50000, type = int,
                    help='Minimum number of tiles a slide must have to include in training. Defaults to 0.')
parser.add_argument('--max_tiles', default = 0, type = int,
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


def create_project(args):
    P = sf.create_project(
        root= args.project_root_dir,
        annotations = args.annotations,
        slides = args.slides_dir,
        dataset_config = args.dataset_config,
        name = args.project_name,
        )
    return P


def extract_tile_dataset(P, args):

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


def train_classifier(args, 
                    dataset, 
                    # train_dataset, 
                    # val_dataset, 
                    P
                ):
    hp = sf.ModelParams(
        tile_px = args.tile_px, 
        tile_um = args.tile_um,
        model = args.model,
        batch_size = args.batch_size,
        epochs = [args.epochs],
    )    
    train_acc, val_loss, val_acc =P.train(outcomes = args.label_column,
                                params=hp,
                                dataset =  dataset, #train_dataset,
                                # val_dataset=val_dataset,
                                # model_type = 'classification',
                                filters= None,
                                filter_blank=None, 
                                # filters={'dataset': ['train', 'eval']},
                                        # 'label': ['subtype_0', 'subtype_1', 'subtype_2','subtype_3', 'subtype_4', 'subtype_5']},
                                min_tiles = args.min_tiles,
                                max_tiles = args.max_tiles,
                                val_strategy= args.val_strategy,
                                val_fraction = args.val_fraction,
                                # val_k_fold = args.val_k_fold,
                                mixed_precision = args.mixed_precision,
                                multi_gpu = args.multi_gpu,
                                checkpoint = args.checkpoint,
                                )    
    return train_acc, val_loss, val_acc


def main(args):
    # Part 01: Create project or load project
    if args.load_project:
        P = sf.load_project(args.project_root_dir)
    else:
        P = create_project(args)    

    # Part 02: Extract tiles from slides    
    if args.load_dataset:
        # Load extracted tile-datset
        dataset = P.dataset(tile_px=args.tile_px, 
                            tile_um=args.tile_um,
                            )
    else:
        dataset = extract_tile_dataset(P, args)   
    # dataset = dataset.filter({"dataset": ['train', 'eval']})
    # train_dataset = dataset.filter({"dataset": ["train"]})
    # val_dataset = dataset.filter({"dataset": ["val"]})

    if args.train_classifier:
        train_acc, val_loss, val_acc = train_classifier(args, dataset, P )
        # train_acc, val_loss, val_acc = train_classifier(args, train_dataset, val_dataset, P )
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
    print("finished training run!")
    end_time = time.time()
    total_time = end_time -start_time
    print('total_time: ',total_time)