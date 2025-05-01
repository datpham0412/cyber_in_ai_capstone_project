from datasets import load_dataset, load_from_disk

# Parameters
text_len = 64
img_subset = "img_dalle"
text_subset = "minigpt4_stage2_text"

# Load datasets from Hugging Face hub
image_dataset = load_dataset("JaineLi/VL-MIA-image", img_subset, split='train')
text_dataset = load_dataset("JaineLi/VL-MIA-text", text_subset, split=f"length_{text_len}")

# Save them locally
image_dataset.save_to_disk(f"{img_subset}")
text_dataset.save_to_disk(f"{text_subset}")
