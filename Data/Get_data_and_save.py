from datasets import load_dataset

# === IMAGE DATASETS ===
image_dataset_name = "JaineLi/VL-MIA-image"
image_subsets = ["img_Flickr", "img_Flickr_2k", "img_Flickr_10k", "img_dalle"]
image_split = "train"

for subset in image_subsets:
    print(f"📥 Loading image subset: {subset}...")
    dataset = load_dataset(image_dataset_name, subset, split=image_split)
    save_path = f"{subset}"
    dataset.save_to_disk(save_path)
    print(f"✅ Saved image subset to: {save_path}/")

# === TEXT DATASETS ===
text_dataset_name = "JaineLi/VL-MIA-text"
text_subset = "minigpt4_stage2_text"
text_lengths = [16, 32, 64]

for length in text_lengths:
    split_name = f"length_{length}"
    print(f"📥 Loading text split: {split_name}...")
    text_dataset = load_dataset(text_dataset_name, text_subset, split=split_name)
    save_path = f"{text_subset}_{split_name}"
    text_dataset.save_to_disk(save_path)
    print(f"✅ Saved text split to: {save_path}/")
