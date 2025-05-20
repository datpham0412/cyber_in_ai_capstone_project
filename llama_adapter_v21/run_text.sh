#!/bin/bash
# ---------------- Slurm directives ----------------
#SBATCH --job-name=vlmia_text_32
#SBATCH --output=vlmia_text_%j.out
#SBATCH --error=vlmia_text_%j.err
#SBATCH --partition=gpu          # <-- real GPU queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
# --------------------------------------------------

# ----- user editable section ----------------------
GPU_ID=0
TEXT_LEN=32
DATASET_SPLIT=llava_v15_gpt_text
LLAMA_WEIGHTS=../llama_weights
ARROW_FOLDER=/fred/oz402/tiend/VL-MIA-text-arrow
PY_SCRIPT=/fred/oz402/tiend/VLLM-MIA/llama_adapter_v21/run_with_text.py
# --------------------------------------------------

module load cuda/11.7.0
module -q load conda
mamba activate llava

echo "[`date`] running ${PY_SCRIPT}"
echo "GPU id  = ${GPU_ID}"
echo "text len= ${TEXT_LEN}"
echo "dataset = ${DATASET_SPLIT}"
echo "arrow   = ${ARROW_FOLDER}"
echo "weights = ${LLAMA_WEIGHTS}"
echo "---------------------------------------------------------------"

python "${PY_SCRIPT}" \
  --gpu_id   "${GPU_ID}" \
  --text_len "${TEXT_LEN}" \
  --dataset  "${DATASET_SPLIT}" \
  --llama_path "${LLAMA_WEIGHTS}" \
  --data_path  "${ARROW_FOLDER}"

echo "[`date`] job finished"
