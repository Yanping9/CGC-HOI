DETR=base python main_linux.py --dataset hicodet --data-root hicodet/ --partitions trainval test \
                         --world-size 1 \
                         --batch-size 1 \
                         --cache \
                         --resume /home/txs/work/yw/training_data/task2/lppvic-detr-r50-vcoco/best.pth \
                         --output-dir /home/txs/work/yw/training_data/task2/lppvic-detr-r50-vcoco