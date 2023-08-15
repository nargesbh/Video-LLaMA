import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from video_llama.common.config import Config
# from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

from tqdm import tqdm
import pandas as pd

#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

from typing import List
import torch
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/video_llama_eval_withaudio.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='vicuna', help="The type of LLM")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    # args = parser.parse_args()
    args, remaining_args = parser.parse_known_args()
    return args, remaining_args

def compute_llama_sentence_embeddings(llama, tokenizer, texts: str | List[str], avg=True, normalize=True):
    inps = tokenizer(texts, return_tensors="pt", padding=True).to(llama.device)

    with torch.no_grad():
        embs = llama(**inps)["last_hidden_state"]

        
    return embs

def main(
    df_dir: str = "/home/ahmadi/video-ir/dataset/1KA/MSRVTT_JSFUSION_test.csv",
    save_dir_path: str = "/home/ahmadi/video-ir/dataset/1KA/llama_txt_embedding_sum",
    First_element: bool = False,
    Max_pooling: bool = False,
    Average_pooling: bool = False,
    Sum: bool = True    
    ):
    # Model Initialization
    print('Initializing Chat')
    args, remaining_args = parse_args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()


    df = pd.read_csv(df_dir)
        
    vid_names = df["video_id"]
    captions = df['sentence']

    tokenizer = model.llama_tokenizer


    for i in tqdm(range(len(captions))):
        print(i)
            
        embeddings = compute_llama_sentence_embeddings(model.llama_model.model, tokenizer, captions[i])
        embeddings = embeddings.float()
        embeddings = embeddings.cpu().detach().numpy()
        
        print(captions[i])
        print(embeddings.shape)
        
        if First_element:
            embeddings = embeddings[0][0]
                
        elif Max_pooling:
            embeddings = np.max(embeddings[0], axis=0)
                
        elif Average_pooling:
            embeddings = np.min(embeddings[0], axis=0)
                
        elif Sum:
            embeddings = np.sum(embeddings[0], axis=0)
        
        print(embeddings.shape)
            
        break
                
                
    np.save(save_dir_path + "/" + vid_names[i] + ".npy", embeddings)
    
