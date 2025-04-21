# VLLM-MIA (Offline Reproduction on OzSTAR HPC)

This repository contains a modified version of the [VL-MIA](https://github.com/LIONS-EPFL/VL-MIA) codebase adapted for **offline execution** on Swinburne’s OzSTAR HPC, which does **not** have internet access.

The original code was developed for internet-enabled environments. The modifications below enable full reproduction in a local and offline HPC setup.

---

## 🛠 Key Modifications

### 1. `get_data.py`

- Downloads the **VL-MIA image dataset** from Hugging Face.
- Saves the dataset locally using `dataset.save_to_disk()`, and load with `load_from_disk()`.

### 2. `load_model.py`

- Downloads and saves all required models locally:
  - **LLaVA model** (`llava-v1.5-7b`)
  - **CLIP Image Processor**
  - **CLIP Vision Model** (`openai/clip-vit-large-patch14-336`)

### 3. `run_with_img.py` and `run_with_text.py`

- Updated all dataset and model paths to point to **local directories**.
- Added a `logger` for better execution tracking.
- Modified `load_image()` to avoid downloading images from URLs:

```python
def load_image(image_file):
    if isinstance(image_file, Image.Image):  
        return image_file.convert("RGB")  
    
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        logger.warning("Internet access detected. Skipping image loading from URL.")
        return None
    else:
        image = Image.open(image_file).convert("RGB")
    
    return image
```

### 4. SLURM Scripts

- `run_img.sh`: Submits image-based MIA job
- `run_text.sh`: Submits text-based MIA job

---

## 🧪 Reproduction Steps

1. **Run data and model setup on a machine with internet:**

- Remember to change the path to your local fodler accordingly.

```bash
python get_data.py     # Downloads and saves dataset locally
python load_model.py   # Downloads and saves models locally
```

2. **Submit SLURM jobs on OzSTAR:**

```bash
sbatch run_img.sh      # For image-based MIA
sbatch run_text.sh     # For text-based MIA
```

---

## 🙋 Maintainer

Modified for offline use by: **Anh Ho**  
Original repo: [LIONS-EPFL/VL-MIA](https://github.com/LIONS-EPFL/VL-MIA)
