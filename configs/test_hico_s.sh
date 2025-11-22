DETR=base python main_linux.py --dataset hicodet --data-root hicodet/ --partitions train2015 test2015 \
                         --world-size 1 \
                         --batch-size 1 \
                         --eval \
                         --resume /home/txs/work/yw/training_data/task2/lppvic-detr-r50-hico/best.pth 