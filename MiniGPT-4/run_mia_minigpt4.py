#!/usr/bin/env python3
import argparse, os, random, logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# — VL-MIA helpers —————————————————————————————————————————————
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from metric_util import get_text_metric, get_meta_metrics, convert, save_output
from eval import fig_fpr_tpr

# — MiniGPT-4 imports ———————————————————————————————————————————
from fastchat import model as fmodel
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.interact import (
    Interact, CONV_VISION_Vicuna0, CONV_VISION_LLama2
)
# register all builders/models/processors/runners/tasks
from minigpt4.datasets.builders import *
from minigpt4.models           import *
from minigpt4.processors       import *
from minigpt4.runners          import *
from minigpt4.tasks            import *

logging.basicConfig(level=logging.ERROR)


def parse_args():
    p = argparse.ArgumentParser(description="MiniGPT-4 VL-MIA Text Attack")
    p.add_argument("--cfg-path",
                   default="eval_configs/minigpt4_eval_local.yaml",
                   help="path to MiniGPT-4 eval config")
    p.add_argument("--gpu_id",  type=int, default=0)
    p.add_argument("--output_dir", type=str, default="text_MIA")
    p.add_argument("--dataset", type=str,
                   default="minigpt4_stage2_text",
                   help="llava_v15_gpt_text or minigpt4_stage2_text")
    p.add_argument("--text_len", type=int, default=32,
                   help="one of 16, 32, 64")
    return p.parse_args()


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_conversation_template(name):
    conv = fmodel.get_conversation_template(name)
    if conv.name == "zero_shot":
        conv.roles = tuple(["### " + r for r in conv.roles])
        conv.sep   = "\n"
    elif conv.name == "llama-2":
        conv.sep2 = conv.sep2.strip()
    return conv


def mod_infer(model, vis_processor, user_message, gpu_id):
    """
    Runs one forward pass, slices out the token logits for the instruction+desc
    and returns the raw metrics dict via get_meta_metrics.
    """
    chat = Interact(model, vis_processor, device=f"cuda:{gpu_id}")

    # use a dummy black image so the pipeline still works
    img = Image.new("RGB", (1024, 1024), color="black")
    img_list = []

    state = CONV_VISION.copy()
    chat.upload_img(img, state, img_list)
    chat.encode_img(img_list)

    # feed an empty user turn so the model is ready
    chat.ask("", state)
    state.append_message(state.roles[1], None)

    # now append our actual text prompt
    state.append_message(user_message, None)

    outputs, input_ids, seg_tokens = chat.get_output_by_emb(
        conv=state, img_list=img_list
    )
    logits = outputs.logits

    # how many tokens to drop for each segment
    right  = seg_tokens[-1].shape[1]
    left   = seg_tokens[0].shape[1]
    descp  = chat.model.llama_tokenizer(
        user_message, return_tensors="pt", add_special_tokens=False
    ).to(chat.device).input_ids

    slice_dict = {
        "inst_desp": slice(-right, None),
        "inst":      slice(-right, -descp.shape[1]),
        "desp":      slice(-descp.shape[1], None),
    }
    sl = slice_dict["inst_desp"]

    # pick out the logits for our target slice
    logits_slice = logits[0, sl, :]
    ids_slice    = input_ids[0][sl]

    probs     = torch.nn.functional.softmax(logits_slice,   dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits_slice, dim=-1)

    return get_meta_metrics(ids_slice, probs, log_probs)


def inference(model, vis_processor, text, ex, gpu_id):
    # do two passes (original and lowercase) for the MIA metrics
    m1 = mod_infer(model, vis_processor, text,        gpu_id)
    m2 = mod_infer(model, vis_processor, text.lower(), gpu_id)

    pred = get_text_metric(
        m1["ppl"],    m1["all_prob"],  m1["loss"],
        m1["entropies"], m1["modified_entropies"],
        m1["max_prob"], m1["probabilities"],
        m1["gap_prob"], m1["renyi_05"], m1["renyi_2"],
        text,          m2["ppl"],
        m1["mod_renyi_05"], m1["mod_renyi_2"],
    )
    ex["pred"] = pred
    return ex


def evaluate_data(model, vis_processor, data, col_name, gpu_id):
    out = []
    for ex in tqdm(data, desc="Running text MIA"):
        out.append(inference(model, vis_processor, ex[col_name], ex, gpu_id))
    return out


if __name__ == "__main__":
    args = parse_args()

    # 1) load your MiniGPT-4 config + model
    cfg = Config(args)
    setup_seeds(cfg.run_cfg.seed + get_rank())

    model_cfg = cfg.model_cfg
    model_cfg.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_cfg.arch)
    model     = model_cls.from_config(model_cfg).to(f"cuda:{args.gpu_id}")

    # choose the right conv template
    CONV_VISION = {
        "pretrain_vicuna0": CONV_VISION_Vicuna0,
        "pretrain_llama2": CONV_VISION_LLama2,
    }[model_cfg.model_type]

    # 2) image processor for MiniGPT-4
    vp_cfg   = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    processor = registry.get_processor_class(vp_cfg.name).from_config(vp_cfg)

    # 3) load the text split
    split_name = f"length_{args.text_len}"
    ds = load_dataset(
        "JaineLi/VL-MIA-text",
        args.dataset,
        split=split_name
    )
    data = convert_huggingface_data_to_list_dic(ds)

    # 4) run the attack
    output_dir = Path(args.output_dir) / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_data(model, processor, data, "input", args.gpu_id)

    # 5) plot ROC + save CSV
    fig_fpr_tpr(results, str(output_dir))
