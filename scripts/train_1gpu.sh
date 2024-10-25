# export nnUNet_N_proc_DA=4

# for v1 nnUNet
# export nnUNet_codebase="/opt/conda/envs/nnunet/lib/python3.10/site-packages/nnunet"
# export nnUNet_raw_data_base="/mnt/disks/data/arahut/ANISHA/atlas_stroke_data/nnUNet_raw_data_base"
# export nnUNet_preprocessed="/mnt/disks/data/arahut/ANISHA/atlas_stroke_data/nnUNet_preprocessed"
# export RESULTS_FOLDER="/mnt/disks/data/arahut/ANISHA/atlas_stroke_data/nnUNet_results"

CONFIG=$1

echo "ANISHA: using CONFIG = $CONFIG"

### unit test
fold=0
echo "run on fold: ${fold}"
# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nnunet_use_progress_bar=1 CUDA_VISIBLE_DEVICES=0 \
          python3 -m torch.distributed.launch --master_port=4322 --nproc_per_node=1 \
          ./train.py --fold=${fold} --config=$CONFIG --resume=''

# for resumption add these
# --resume='local_latest' --continue_training
