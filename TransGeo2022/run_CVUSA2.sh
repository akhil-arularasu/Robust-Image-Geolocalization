#!/usr/bin/env bash
## python -u addingNoiseCluster.py
# python -u trainTemp.py --lr 0.0001 --batch-size 32 --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100 --resume ./result_newModel/model_best.pth.tar --save_path ./result_newModel --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5 --evaluate

## for i in {0..17}; do
##   echo "Iteration: $i"
##    python -u train.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100 --resume ./PretrainedModel/result/model_best.pth.tar --save_path ./result_all2 --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5 --sat_res 320 --crop --evaluate --noiseIndex "$i"
## done

## python -u train.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 50 --resume ./result/checkpoint.pth.tar --save_path ./result --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5 --sat_res 320 --crop -- evaluate

python -u trainLvl4Gaussian.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100 --resume ./result_lvl4C+NGaussian/model_best.pth.tar --save_path ./result_lvl4C+NGaussian --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5 --evaluate

###these two:::::
##python -u trainTemp.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100 --resume ./result_randomNoiseSeveritylvl2/model_best.pth.tar --save_path ./result_randomNoiseSeveritylvl2 --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5
## python -u train.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100 --save_path ./result_optimizedRandomNoise --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5


# python -u trainingImagesWNoise.py