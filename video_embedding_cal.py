import argparse
from video_llama.common.config import Config
from video_llama.common.registry import registry
from Zconversation_video import Chat
import os
import torch
import fire
import numpy as np
from tqdm import tqdm 
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


def main (
        video_dir_path: str = "/home/ahmadi/video-ir/dataset/1KA/test_1k_compress/", 
        save_dir_path: str = "/home/ahmadi/video-ir/dataset/1KA/llama_vid_embedding_sum/",
        First_element: bool = False,
        Max_pooling: bool = False,
        Average_pooling: bool = False,
        Sum: bool = True,
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


    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    
    for path in tqdm(os.listdir(video_dir_path)):
        img_list = []
        list_emb = chat.upload_video_without_audio(video_dir_path+path, img_list)
        list_emb = list_emb[0][0].cpu().detach().numpy()
        
        
        if First_element:
            list_emb = list_emb[0]
            
        elif Max_pooling:
            list_emb = np.max(list_emb, axis=0)
            
        elif Average_pooling:
            list_emb = np.mean(list_emb, axis=0)
            
        elif Sum:
            list_emb = np.sum(list_emb, axis=0)
            
        np.save(save_dir_path + "/" + path[:-4] + ".npy", list_emb)
    

if __name__ == "__main__":
    fire.Fire(main)


