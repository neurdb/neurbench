# D-LQO


## Env Config

```bash
# Create conda env from yml (recommended)
conda env create -f environment_moqoe.yml
conda activate dlqo

# Set Python path (run from repo root)
export PYTHONPATH=$PYTHONPATH:./src
```

## Dataset

Generate datasets for training and testing

```bash
# IMDB
python ./src/data_collector/router_dataset_custome.py 
```

We include the datasets to train the **expert routing network** here

```bash
cd ./experiment_result/datasets 
```

## Training

```bash
python ./src/expert_router/router_offline_pretrain.py --epochs 100 --embedding_dim 256 --lr 0.0005 --batch_size 16 --step_size 500 --gamma 0.5 --single_exp all_train --threshold 0.01 --alpha 0.5 --loss_gamma 2 --num_self_attn_layers 2
```

## Inference

```bash
# for imdb
python ./src/expert_router/router_inference.py --dataset imdb
```

