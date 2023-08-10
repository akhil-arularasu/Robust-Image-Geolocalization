#!/usr/bin/env bash
## python -u addingNoiseCluster.py
## python -u trainTemp.py --lr 0.0001 --batch-size 32 --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100 --resume ./result_newModel/model_best.pth.tar --save_path ./result_newModel --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5 --evaluate

# Iterate over the range
## for i in $(seq 0 17)
##  do
##  echo "Iteration: $i"
##  python -u trainTemp.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100 --resume ./result_lvl4C+NRandom2/model_best.pth.tar --save_path ./result_lvl4C+NRandom2 --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5 --evaluate --noiseIndex "$i"
##  done

## python -u train.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 50 --resume ./result/checkpoint.pth.tar --save_path ./result --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5 --sat_res 320 --crop --evaluate
##python -u train.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100 --resume ./PretrainedModel/result/model_best.pth.tar --save_path ./result_all2 --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5 --sat_res 320 --crop --evaluate --noiseIndex "$i"

## [training random noises lvl4] python -u trainTemp4.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100 --save_path ./result_lvl4C+NRandom2 --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5

## [eval random noises lvl4] 
python -u trainLvl4Random.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100  --resume ./result_lvl4C+NRandom/model_best.pth.tar --save_path ./result_lvl4C+NRandom --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5 --evaluate


###these two:::::
##python -u trainTemp.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100 --resume ./result_randomNoiseSeveritylvl2/model_best.pth.tar --save_path ./result_randomNoiseSeveritylvl2 --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5
## python -u train.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100 --save_path ./result_optimizedRandomNoise --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 1000 --asam --rho 2.5


## python -u trainingImagesWNoise.py