from datasets import load_dataset, load_from_disk

save_path_img = "/fred/oz402/aho/VLLM-MIA/Data/img_Flickr"
save_path_text = "/fred/oz402/aho/VLLM-MIA/Data/llava_v15_gpt_text"

# image #
# img_subset = "img_Flickr" # or img_dalle
# image_dataset = load_dataset("JaineLi/VL-MIA-image", img_subset, split='train')
# image_dataset.save_to_disk(save_path_img)

# -------------------------------------------------------------------
# text #
# text_len = 64 # 16,32,64
# text_subset = "llava_v15_gpt_text" # or minigpt4_stage2_text
# text_dataset = load_dataset("JaineLi/VL-MIA-text", text_subset, split=f"length_{text_len}")
# text_dataset.save_to_disk(save_path_text)

# --------------------------------------------------------------------
# check data #

dataset = load_from_disk(save_path_img)
print(dataset)
print(dataset[5])