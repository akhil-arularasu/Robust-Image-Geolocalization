#!/usr/bin/env bash
python -u addingNoiseCluster.py
## python -u train.py --lr 0.0001 --batch-size 32 --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100 --resume ./result/model_best.pth.tar --save_path ./result_all2 --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5 --evaluate
####use this one####python -u train.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100 --resume ./PretrainedModel/result/model_best.pth.tar --save_path ./result_all2 --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5 --sat_res 320 --crop --evaluate
## python -u train.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 50 --resume ./result/checkpoint.pth.tar --save_path ./result --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5 --sat_res 320 --crop -- evaluate

## you need: --resume ./model_best.pth.tar
## also need: --evaluate