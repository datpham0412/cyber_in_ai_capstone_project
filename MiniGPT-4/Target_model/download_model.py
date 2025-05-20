from transformers import AutoTokenizer, AutoModelForCausalLM

# Set your target directory
target_dir = "/fred/oz402/nhnguyen/Model_PJ/VLLM-MIA/MiniGPT-4/Target_model"  # CHANGE THIS

model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True, cache_dir=target_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=True, cache_dir=target_dir)
