python main.py \
-t \
--base configs/objaverse34.yaml \
--logdir /home/user/free3d_work/test/logs \
--name test \
--gpus 0,1 \
--scale_lr False \
--num_nodes 1 \
--seed 42 \
--check_val_every_n_epoch 10 \
--finetune_from /home/user/free3d_models/000005.ckpt
# --finetune_from /work/cxzheng/code/zero123_old/zero123/sd-image-conditioned-v2.ckpt
