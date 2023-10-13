"""
Conversation prompt template of Video-LLaMA.
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/conversation/conversation.py 
"""
import argparse
import time
from PIL import Image
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
import os
from video_llama.common.registry import registry
from video_llama.processors.video_processor import ToTHWC,ToUint8,load_video
from video_llama.processors import Blip2ImageEvalProcessor
            
from video_llama.models.ImageBind.data import load_and_transform_audio_data

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        self.image_vis_processor = Blip2ImageEvalProcessor()
        # stop_words_ids = [torch.tensor([835]).to(self.device),
        #                   torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        # self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])


    
    def upload_video(self, video_path, img_list):

        msg = ""
        if isinstance(video_path, str):  # is a video path
            ext = os.path.splitext(video_path)[-1].lower()
            print(video_path)
            # image = self.vis_processor(image).unsqueeze(0).to(self.device)
            video, msg = load_video(
                video_path=video_path,
                n_frms=8,
                height=224,
                width=224,
                sampling ="uniform", return_msg = True
            )
            video = self.vis_processor.transform(video)
            video = video.unsqueeze(0).to(self.device)
            # print(image)
        else:
            raise NotImplementedError
        
        try:
            audio_flag = 1
            audio = load_and_transform_audio_data([video_path],"cpu",  clips_per_video=8)
            audio = audio.to(self.device)
        except :
            print('no audio is found')
            audio_flag = 0
        finally:
            if audio_flag == 1:
                # image_emb, _ = self.model.encode_videoQformer_audiovideo(video,audio)
                image_emb, _ = self.model.encode_videoQformer_visual(video)
                audio_emb,_  = self.model.encode_audioQformer(audio)
                img_list.append(image_emb)
                img_list.append(audio_emb)

            else:  # only vison no audio
                # conv.system = "You can understand the video that the user provides. Follow the instructions carefully and explain your answers in detail."
                image_emb, _ = self.model.encode_videoQformer_visual(video)
                img_list.append(image_emb)
            return img_list

    def upload_video_without_audio(self, video_path, img_list):
        msg = ""
        if isinstance(video_path, str):  # is a video path
            ext = os.path.splitext(video_path)[-1].lower()
            print(video_path)
            # image = self.vis_processor(image).unsqueeze(0).to(self.device)
            video, msg = load_video(
                video_path=video_path,
                n_frms=8,
                height=224,
                width=224,
                sampling ="uniform", return_msg = True
            )
            video = self.vis_processor.transform(video)
            video = video.unsqueeze(0).to(self.device)
            # print(image)
        else:
            raise NotImplementedError
        
        
        # conv.system = "You can understand the video that the user provides.  Follow the instructions carefully and explain your answers in detail."
        image_emb, _ = self.model.encode_videoQformer_visual(video)
        img_list.append(image_emb)
        return img_list


if __name__ =='__main__':
    video_path = '/mnt/workspace/videoGPT/Video-LLaMA/examples/applausing.mp4'
    # import torch.classes.torchaudio.ffmpeg_StreamReader
    # ffmpeg_StreamReader(video_path)
    load_and_transform_audio_data([video_path],"cpu",  clips_per_video=8)
    
    
    


