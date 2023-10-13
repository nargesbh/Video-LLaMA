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

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        self.image_vis_processor = Blip2ImageEvalProcessor()
        # stop_words_ids = [torch.tensor([835]).to(self.device),
        #                   torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        # self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    
    def prepare_inputs_for_generation(
        self, input_ids, query_embeds=None, past_key_values=None, attention_mask=None, inputs_embeds=None, kwargs = {'use_cache': True}
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        n = inputs_embeds.shape[1]
        position_ids = torch.arange(n).unsqueeze(0).to(inputs_embeds.device)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
                query_embeds = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "query_embeds": query_embeds,
                "past_key_values": past_key_values,
                "use_cache": kwargs['use_cache'],
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    
    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)
        
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        
        print(begin_idx)

        embs = embs[:, begin_idx:]
        if conv.sep =="###":
            stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
            
        
        input_ids = torch.tensor([[1]], device=embs.device)
        
        n = embs.shape[1]
        position_ids = torch.arange(n).unsqueeze(0).to(embs.device)
        attention_mask = torch.ones((1, n), dtype=torch.long).to(embs.device)
        
        model_inputs = self.prepare_inputs_for_generation(input_ids, inputs_embeds = embs, attention_mask =attention_mask )
        
        past_key_values = None
        query_embeds = None
        use_cache = True
        output_attentions = False
        output_hidden_states = False
        return_dict = True
        
        outputs = self.model.llama_model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds = model_inputs['inputs_embeds'],
            query_embeds=query_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
                )

        hidden_states = outputs[0]
        
        return hidden_states
        



    def upload_video_without_audio(self, video_path, conv, img_list):
        msg = ""
        if isinstance(video_path, str):  # is a video path
            ext = os.path.splitext(video_path)[-1].lower()
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
        conv.append_message(conv.roles[0], "<Video><ImageHere></Video> "+ msg)
        return "Received."


    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
    


if __name__ =='__main__':
    video_path = '/mnt/workspace/videoGPT/Video-LLaMA/examples/applausing.mp4'
    # import torch.classes.torchaudio.ffmpeg_StreamReader
    # ffmpeg_StreamReader(video_path)
    load_and_transform_audio_data([video_path],"cpu",  clips_per_video=8)