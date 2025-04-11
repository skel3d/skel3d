module load singularity
fusermount -u /project/c_mvcondif/objaverse_skel
fuse2fs -o fakeroot,auto_unmount,ro /project/c_mvcondif/objaverse_skel.img /project/c_mvcondif/objaverse_skel
export WANDB_API_KEY=cac4802235eab1dcc55870599944f661d7697032
#export XDG_CACHE_HOME=/cache/
singularity exec --nv -B /project/c_mvcondif/venvs/:/venvs/ \
    -B /project/c_mvcondif/objaverse_skel/:/objaverse/ \
    -B /project/c_mvcondif/models/:/models/ \
    -B /project/c_mvcondif/logs/:/logs/ \
    -B /project/c_mvcondif/outputs/:/outputs/ \
      /project/c_mvcondif/containers/free3d.sif \
      python main.py \
        -t \
        --base configs/objaverse_skel.yaml \
        --logdir /logs \
        --name fa_test \
        --gpus 0,1,2,3 \
        --scale_lr False \
        --num_nodes 1 \
        --seed 42 \
        --check_val_every_n_epoch 10 \
        --finetune_from /models/000005.ckpt

fusermount -u /project/c_mvcondif/objaverse_skel
    # --finetune_from /work/cxzheng/code/zero123_old/zero123/sd-image-conditioned-v2.ckpt 244/23911
