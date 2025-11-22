DETR=base python main_linux.py --dataset hicodet --data-root hicodet/ --partitions train2015 test2015 \
                         --pretrained ./checkpoints/detr-r50-hicodet.pth \
                         --output-dir /home/txs/work/yw/training_data/task2/hico \
                         --world-size 2 --epochs 30 --batch-size 16