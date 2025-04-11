#!/bin/bash
#SBATCH -A c_mvcondif
#SBATCH --time=1-00:00:00
#SBATCH --job-name=fothi_nvs_skel3d_coord
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=email@example.com
module load singularity


export WANDB_DATA_DIR=/wandb_data


singularity exec --nv --env WANDB_API_KEY="api_key" -B /project/c_mvcondif/venvs/:/venvs/ \
-B /scratch/c_mvcondif/:/objaverse/ \
-B /project/c_mvcondif/models/:/models/ \
-B /project/c_mvcondif/logs/:/logs/ \
-B /scratch/c_mvcondif/wandb_data/:/wandb_data/ \
-B /project/c_mvcondif/outputs/:/outputs/ \
/project/c_mvcondif/containers/free3d.sif \
python main.py \
-t \
--base configs/objaverse_hdf5_skel_coord.yaml \
--logdir /logs \
--name skel3d_ai_gpu8_hdf_coord \
--gpus 0,1,2,3,4,5,6,7 \
--scale_lr False \
--num_nodes 1 \
--seed 42 \
--check_val_every_n_epoch 1 \
--finetune_from /models/000005.ckpt






for checkpoint in "${checkpoints[@]}"; do
    output="hdf5_skel3d_warp_${checkpoint}"
    
    singularity exec --nv -B /project/c_mvcondif/venvs/:/venvs/ \
    -B /scratch/c_mvcondif/:/objaverse/ \
    -B /project/c_mvcondif/models/:/models/ \
    -B /project/c_mvcondif/logs/:/logs/ \
    -B /scratch/c_mvcondif/wandb_data/:/wandb_data/ \
    -B /project/c_mvcondif/outputs/:/outputs/ \
    /project/c_mvcondif/containers/free3d.sif \
    python batch_test.py \
    --resume /logs/$model_path/checkpoints/$checkpoint \
    --config configs/objaverse_hdf5_skel_coord.yaml \
    --save_path /outputs/$output
    
    singularity exec --nv -B /project/c_mvcondif/venvs/:/venvs/ \
    -B /scratch/c_mvcondif/:/objaverse/ \
    -B /project/c_mvcondif/models/:/models/ \
    -B /project/c_mvcondif/logs/:/logs/ \
    -B /project/c_mvcondif/outputs/:/outputs/ \
    /project/c_mvcondif/containers/free3d.sif \
    python evaluations/evaluation.py --gt_path /outputs/$output/inputs/ --g_path /outputs/$output/samples_cfg_scale_3.00/ --report_path skel3d_warp.csv
done

