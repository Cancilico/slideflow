conda install -y -c conda-forge ipykernel=6.29.4
python -m ipykernel install --user --name slideflow --display-name="slideflow"
conda install -y -c conda-forge mlflow=1.26.1
conda install -y -c conda-forge optuna=4.0.0

pip install slideflow-gpl==0.0.2
pip install slideflow-noncommercial==0.0.2
pip install nystrom-attention==0.0.9
pip install typing-extensions==4.12.2