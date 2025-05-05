**VL‑MIA from zero to results – plain step‑by‑step** 


---



### A. Set three absolute paths once 



```bash
# folder that will hold 7 B weights + tokenizer
export VLMIA_LLAMA_DIR=/fred/oz402/tiend/llama_weights

# on‑disk Arrow datasets
export VLMIA_TEXT_ARROW=/fred/oz402/tiend/VL-MIA-text-arrow
export VLMIA_IMG_ARROW=/fred/oz402/tiend/VL-MIA-image-arrow      # may skip images

# where you clone the repo
export VLMIA_ROOT=/fred/oz402/tiend/VLLM-MIA
```



---



### 1. Clone and create the Python environment 



```bash
git clone https://github.com/LIONS-EPFL/VL-MIA.git  "$VLMIA_ROOT"
cd   "$VLMIA_ROOT"

mamba create -n llava python=3.10 -y
mamba activate llava
pip install -r llama_adapter_v21/requirements.txt       # torch 2.1, timm, clip‑anytorch …
```

(no internet?  build wheels elsewhere and install with `pip --no-index --find-links wheels`).


---



### 2. Download a public base model (OpenLLaMA‑7B, MIT licence) 



```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="openlm-research/open_llama_7b",
  allow_patterns=["tokenizer.model","config.json","pytorch_model-*.bin"],
  local_dir="$VLMIA_LLAMA_DIR", local_dir_use_symlinks=False)
PY

mkdir -p "$VLMIA_LLAMA_DIR/7B"
mv  "$VLMIA_LLAMA_DIR/config.json" "$VLMIA_LLAMA_DIR/7B/params.json"
i=0; for f in "$VLMIA_LLAMA_DIR"/pytorch_model-*.bin; do
      printf -v new "$VLMIA_LLAMA_DIR/7B/consolidated.%02d.pth" $i
      mv "$f" "$new"; ((i++)); done
```

Edit `$VLMIA_LLAMA_DIR/7B/params.json` so it contains only 6 keys:


```json
{"dim":4096,"n_layers":32,"n_heads":32,"vocab_size":32000,
 "multiple_of":256,"norm_eps":1e-5}
```

*(Meta LLaMA‑2 works too – same directory layout.)*


---



### 3. Cache the datasets once (do it on a machine with internet) 

**Text split** 


```java
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("JaineLi/VL-MIA-text", "llava_v15_gpt_text")
ds.save_to_disk("$VLMIA_TEXT_ARROW")
PY
```

**Image split (optional)** 


```java
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("JaineLi/VL-MIA-image", "img_Flickr")
ds.save_to_disk("$VLMIA_IMG_ARROW")
PY
```


Copy the resulting folders to any offline compute node.



---



### 4. Quick interactive check (text) 



```bash
sinteractive --partition=gpu --gres=gpu:1 --mem=32G --time=02:00:00
mamba activate llava
cd "$VLMIA_ROOT/llama_adapter_v21"

python run_with_text.py \
  --gpu_id 0 --text_len 32 \
  --dataset  llava_v15_gpt_text \
  --llama_path "$VLMIA_LLAMA_DIR" \
  --data_path  "$VLMIA_TEXT_ARROW"
```


You should see:

 
- a long list of **Trainable param:**  lines (bias + norm only)
 
- progress bar over 600 samples
 
- attack metrics in console
 
- `text_MIA/length_32/auc.png` and `auc.txt`.



---



### 5. Batch scripts for Slurm 


#### 5.1 Text – save everything to one log 

`run_text.sh`


```bash
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

```

Submit with `sbatch run_text.sh`.

#### 5.2 Image – Flickr split 

`run_img.sh`


```bash
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

```

Submit with `sbatch run_img.sh`.


---



### 6. Where to look afterwards 



```bash
llama_adapter_v21/
 ├─ text_MIA/length_32/auc.png  • ROC of main metric
 │                              • auc.txt table of all attacks
 └─ image_MIA/img_Flickr/gen_32_tokens/
      inst/auc.png, inst_desp/…, desp/…
slurm-<jobid>.out  • stdout (params list, progress, metrics)
slurm-<jobid>.err  • stderr
```



---



### 7. Common issues 

 
- **OOM killed immediately**  → request at least 12 GB GPU, or lower

`--text_len` / `--num_gen_token`.
 
- **KeyError: split**  → ensure the Arrow folder really contains a

`train/` split or the named length split.
 
- **No internet on node**  → keep `$VLMIA_TEXT_ARROW` and

`$VLMIA_IMG_ARROW`; the scripts fall back to Hugging Face only if the

local copy is missing.

Once these steps work you can rerun anytime by changing only the

Slurm header or dataset split – everything else is reproducible.