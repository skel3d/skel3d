#!/bin/bash
#SBATCH -A c_mvcondif
#SBATCH --time=01:00:00
#SBATCH --job-name=fothi_nvs_eval
#SBATCH --output=slurm_out_val/%j.out
#SBATCH --error=slurm_out_val/%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-cpu=40G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=email@example.com
module load singularity

export WANDB_DATA_DIR=/wandb_data


model_path='/2025-02-05T19-53-58_skel3d_ai_gpu8_hdf/'

checkpoints=($(ls /project/c_mvcondif/logs/$model_path/checkpoints/epoch=*.ckpt | xargs -n 1 basename))
if [ ${#checkpoints[@]} -eq 0 ]; then
    echo "No checkpoints found in /project/c_mvcondif/logs/$model_path/checkpoints/"
    exit 1
fi

checkpoints=('epoch=000068.ckpt')

for checkpoint in "${checkpoints[@]}"; do
    output="hdf5_skel3d_${checkpoint}"
    
    singularity exec --nv -B /project/c_mvcondif/venvs/:/venvs/ \
    -B /scratch/c_mvcondif/:/objaverse/ \
    -B /project/c_mvcondif/models/:/models/ \
    -B /project/c_mvcondif/logs/:/logs/ \
    -B /scratch/c_mvcondif/wandb_data/:/wandb_data/ \
    -B /project/c_mvcondif/outputs/:/outputs/ \
    /project/c_mvcondif/containers/free3d.sif \
    python batch_test.py \
    --resume /logs/$model_path/checkpoints/$checkpoint \
    --config configs/objaverse_hdf5_skel.yaml \
    --save_path /outputs/$output
    
    singularity exec --nv -B /project/c_mvcondif/venvs/:/venvs/ \
    -B /scratch/c_mvcondif/:/objaverse/ \
    -B /project/c_mvcondif/models/:/models/ \
    -B /project/c_mvcondif/logs/:/logs/ \
    -B /project/c_mvcondif/outputs/:/outputs/ \
    /project/c_mvcondif/containers/free3d.sif \
    python evaluations/evaluation.py --gt_path /outputs/$output/inputs/ --g_path /outputs/$output/samples_cfg_scale_3.00/ --report_path skel3d.csv
done


for checkpoint in "${checkpoints[@]}"; do
    output="hdf5_skel3d_${checkpoint}"
    
    singularity exec --nv -B /project/c_mvcondif/venvs/:/venvs/ \
    -B /scratch/c_mvcondif/:/objaverse/ \
    -B /project/c_mvcondif/models/:/models/ \
    -B /project/c_mvcondif/logs/:/logs/ \
    -B /scratch/c_mvcondif/wandb_data/:/wandb_data/ \
    -B /project/c_mvcondif/outputs/:/outputs/ \
    /project/c_mvcondif/containers/free3d.sif \
    python batch_test.py \
    --resume /logs/$model_path/checkpoints/$checkpoint \
    --config configs/objaverse_hdf5_skel.yaml \
    --save_path /outputs/$output
    
    singularity exec --nv -B /project/c_mvcondif/venvs/:/venvs/ \
    -B /scratch/c_mvcondif/:/objaverse/ \
    -B /project/c_mvcondif/models/:/models/ \
    -B /project/c_mvcondif/logs/:/logs/ \
    -B /project/c_mvcondif/outputs/:/outputs/ \
    /project/c_mvcondif/containers/free3d.sif \
    python evaluations/evaluation.py --gt_path /outputs/$output/inputs/ --g_path /outputs/$output/samples_cfg_scale_3.00/ --report_path skel3d.csv
done

