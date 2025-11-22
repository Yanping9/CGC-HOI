DETR=base python main_linux.py --dataset vcoco --data-root vcoco/ --partitions trainval test \
                         --pretrained ./checkpoints/detr-r50-vcoco.pth \
                         --output-dir /home/txs/work/yw/training_data/task2/vcoco \
                         --world-size 2 --epochs 30 --batch-size 16