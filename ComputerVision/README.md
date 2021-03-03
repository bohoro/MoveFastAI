It uses a combination of Hydra, PyTorch Lightning, SubmitIt and other tools;

## Create a conda env:

    conda create -n my_project python=3.8
    conda activate my_project
    pip install -r requirements.txt

## Local training:

    ./train.py +mode=local

## Single node run on SLURM:

    ./train.py -m +mode=single_node

## Multi node run on SLURM:

    ./train.py -m +mode=multi_node

## Param sweep, single node on SLURM:

This launches two jobs, each with different learning rate training on a single node:

        ./train.py -m module.learning_rate=1e-3,1e-4 +mode=single_node

## Param sweep, multi-node on SLURM:

This launches two jobs, each with different learning rate training on multiple nodes:

        ./train.py -m module.learning_rate=1e-3,1e-4 +mode=multi_node
