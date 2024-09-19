import slideflow as sf
import slideflow_gpl as sfg
from slideflow_gpl.extractors import ctranspath
from slideflow_gpl.clam import CLAMModelConfig 
from slideflow_gpl.clam import clam_sb
import slideflow_noncommercial as sfn
from slideflow.mil import mil_config
from slideflow.mil import train_mil
import time
import json
import os
import glob

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize



def main():
    #NOTE: To perform evaluation, change the annotation file (.csv) in setting.json
    # B. Loop over all the MIL models to evaluate the test set and record performace metrics
    project_path = r'/mnt/d/github/slideflow/panda_project'

    # # Load extracted tile-datset
    P = sf.load_project(project_path)  
    dataset = P.dataset(tile_px=244, tile_um=112)

    bag_names = ['bags_ctranspath','bags_histossl','bags_plip',
                'bags_retccl','bags_resnet50_imagenet',
                'bags_alexnet_imagenet', 'bags_inception_imagenet', 
                'bags_resnet18_imagenet', 'bags_resnext50_32x4d_imagenet', 
                'bags_wide_resnet50_2_imagenet']
    mil_models =  ['attention_mil','clam_sb', 'clam_mb', 'mil_fc_mc', 'transmil']

    for bag in bag_names:
    # Loop over each model type
        for model in mil_models:
            print(f"Training model {model} with bags from {bag}")
            model_dir = os.path.join(project_path,f'model_{model}_{bag}')
            print('model_dir:',model_dir)
            mil_model_files = glob.glob(f'{model_dir}/**/models/**.pth',  recursive = True)
            # print(mil_model_files)
            for mil_model_file in mil_model_files:
                print(mil_model_file)
                save_path = mil_model_file.split('models')[0]
                # print(save_path)

        #         # Define configuration common to all models
                config_args = {
                    'model': model,
                    'lr': 1e-4,
                    'batch_size': 32,
                    'epochs': 50,
                    'fit_one_cycle': True,
                }

                if model in ['clam_sb', 'clam_mb', 'mil_fc_mc']:
                    config_args['B'] = 45

                # Configure the MIL model for the current combination
                config = mil_config(**config_args)

                # Evaluate a saved MIL model
                df_test_pred = P.evaluate_mil(
                    config=config,
                    model= mil_model_file,
                    outcomes='label',
                    dataset=dataset,        
                    bags=f'panda_project/feature_bags/{bag}/',
                )
                df_test_pred.to_csv(f'{save_path}/test_prediction.csv')

                #performance metrics
                # Converting prediction probabilities into class predictions
                predicted_probabilities = df_test_pred.filter(regex='y_pred').values
                y_pred = np.argmax(predicted_probabilities, axis=1)
                y_true = df_test_pred['y_true'].values

                # For ROC AUC calculation, we need binary format of y_true
                n_classes = predicted_probabilities.shape[1]
                y_true_binary = label_binarize(y_true, classes=range(n_classes))

                overall_accuracy = accuracy_score(y_true, y_pred)
                print("Overall Accuracy:", overall_accuracy)


                weighted_f1 = f1_score(y_true, y_pred, average='weighted')  # Weighted average F1 score
                print("Overall F1 Score:", weighted_f1)
                f1_class = f1_score(y_true, y_pred, average=None)  # Per-class F1 score
                print("F1 Score per Class:", f1_class)
                clf_report = classification_report(y_true, y_pred, output_dict= True)
                print("Classification Report:\n", clf_report)

                roc_auc = roc_auc_score(y_true_binary, predicted_probabilities, multi_class='ovr')
                print("Overall ROC AUC Score:", roc_auc)

                # Store all metrics in a dictionary
                metrics = {
                    "Test Accuracy": overall_accuracy,
                    "Weighted F1 Score": weighted_f1,
                    "Per-class F1 Score": f1_class.tolist(),  # Convert numpy array to list for JSON serialization
                    "Classification Report": clf_report,
                    "Test ROC AUC Score": roc_auc
                }

                output_file = f'{save_path}/test_metrics_output.json'
                output_file_clf_report = f'{save_path}/test_clf_report.json'
                # Save metrics to a JSON file
                with open(output_file_clf_report, 'w') as f:
                    json.dump(clf_report, f, indent=4)

                with open(output_file, 'w') as f:
                    json.dump(metrics, f, indent=4)                    

                print("Metrics saved to", output_file)



if __name__ == "__main__":
    main()                