model_path='/2025-02-05T18-24-08_fre3d_ai_gpu8_hdf/'

checkpoints=($(ls /project/c_mvcondif/logs/$model_path/checkpoints/epoch=*.ckpt))

if [ ${#checkpoints[@]} -eq 0 ]; then
    echo "No checkpoints found in /project/c_mvcondif/logs/$model_path/checkpoints/"
    exit 1
else
    echo "Found checkpoints:"
    for checkpoint in "${checkpoints[@]}"; 
    do
        echo "$checkpoint"
    done
fi