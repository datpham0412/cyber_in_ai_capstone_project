#!/bin/bash
# ---------------- Slurm directives ----------------
#SBATCH --job-name=vlmia_img_Flickr
#SBATCH --output=vlmia_img_%j.out      # stdout   → vlmia_img_<jobid>.out
#SBATCH --error=vlmia_img_%j.err       # stderr   → vlmia_img_<jobid>.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
# --------------------------------------------------

# ====== user‑editable section =========================================
GPU_ID=0
NUM_GEN=32
DATASET_SPLIT=img_Flickr                     # argument to --dataset
LLAMA_WEIGHTS=../llama_weights               # folder with 7B weights
ARROW_FOLDER=/fred/oz402/tiend/VL-MIA-image-arrow   # local on‑disk dataset
PY_SCRIPT=/fred/oz402/tiend/VLLM-MIA/llama_adapter_v21/run_with_img.py
# =====================================================================

module load cuda/11.7.0
module -q load conda
mamba activate llava

echo "[`date`] running ${PY_SCRIPT}"
echo "GPU id     = ${GPU_ID}"
echo "num_gen    = ${NUM_GEN}"
echo "dataset    = ${DATASET_SPLIT}"
echo "arrow path = ${ARROW_FOLDER}"
echo "weights    = ${LLAMA_WEIGHTS}"
echo "---------------------------------------------------------------"

python "${PY_SCRIPT}" \
  --gpu_id        "${GPU_ID}" \
  --num_gen_token "${NUM_GEN}" \
  --dataset       "${DATASET_SPLIT}" \
  --llama_path    "${LLAMA_WEIGHTS}" \
  --data_path     "${ARROW_FOLDER}"

echo "[`date`] job finished"
