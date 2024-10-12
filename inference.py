import io
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu    # pip install decord
import re
import time
import pickle

MODEL_PATH = "openbmb/MiniCPM-V-2_6-int4"

# #最大帧数
# MAX_NUM_FRAMES = 360
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

parser = argparse.ArgumentParser(description="CogVLM2-Video CLI Demo")
parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=4)
args = parser.parse_args([])


#视频编码
def encode_video(video_data):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    # bridge.set_bridge('torch')
    f=open(f"/vl/{int(time.time())}.pkl","wb+")
    mp4_stream = video_data
    pickle.dump(video_data,f)
    # print(mp4_stream)
    vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    vlength =  len(frame_idx)
    # if len(frame_idx) > MAX_NUM_FRAMES:
        # frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames,vlength



def extract_res(resp):
    pattern = r"(\d+)"
    type_ids = re.findall(pattern, resp)
    return int(type_ids[0])

class llm():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            # padding_side="left"
        )
        # Load the model
        if args.quant == 4:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="cuda:0",
                torch_dtype=TORCH_TYPE,
                trust_remote_code=True,
                cache_dir="/model",resume_download=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=TORCH_TYPE,
                ),
                low_cpu_mem_usage=True
            ).eval()
        elif args.quant == 8:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=TORCH_TYPE,
                trust_remote_code=True,
                cache_dir="/vl/model",
                quantization_config=BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_4bit_compute_dtype=TORCH_TYPE,
                ),
                low_cpu_mem_usage=True
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=TORCH_TYPE,
                cache_dir="/vl/model",
                trust_remote_code=True
            ).eval().to(DEVICE)
    
    def model_predict(self,prompt, video_data, temperature):
        strategy = 'chat'

        history = []
        query = prompt
        params = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 2,
            "do_sample": False,
            "top_p": 0.5,
            "temperature": temperature,
            "use_image_id":False,
            "max_slice_nums":1,
            "do_sample":True
        }
        msgs = [{'role': 'user', 'content': video_data + [query]}]

        # Set decode params for video
        answer = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            **params
        )
        return answer
    
    def vllm(self, prompt, frames_list, vlength, threshold=10, skipframe=10):
        l = len(frames_list)
        per_frame = vlength / l  # 每帧对应的秒数
        step = int(skipframe)  # 每10秒检测一次
        yes_segments = []  # 存储YES段的时间区间
        current_segment_start = None  # 记录当前YES段的开始时间
        temperature = 0.5
        lp = 0
        f = open("./progress.txt","w+")
        f.write(str(0))
        f.close()         
        while lp < l:
            prompt_text = f""" 
            下面有任务你需要回答：
            task:{prompt}
            只需要按下面的格式输出，不需要输出其它内容：
            {{
            'result':"Yes or No"
            }}
            """
            tmp = self.model_predict(prompt_text, [frames_list[lp]], temperature)
            f = open("./progress.txt","w+")
            f.write(str(lp/l))
            f.close()
            if "Yes" in tmp or "yes" in tmp:
                # 向前回溯，找到 "Yes" 的开始
                if current_segment_start is None:
                    current_segment_start = lp * per_frame
                    # 回溯检查前面的帧
                    for i in range(lp - 1, max(lp - step, 0), -1):
                        tmp_back = self.model_predict(prompt_text, [frames_list[i]], temperature)
                        if "Yes" in tmp_back or "yes" in tmp_back:
                            print(i)
                            current_segment_start = i * per_frame
                        else:
                            break

                # 向后扩展，直到找到 "No"
                while lp < l:
                    tmp = self.model_predict(prompt_text, [frames_list[lp]], temperature)
                    if "Yes" in tmp or "yes" in tmp:
                        lp += 1  # 继续向前扩展
                    else:
                        break
                if lp * per_frame - current_segment_start>threshold:
                    # 记录YES段的结束时间
                    yes_segments.append((current_segment_start, lp * per_frame))
                current_segment_start = None  # 重置段开始时间
            else:
                lp += step  # 按每10秒步进
            
            torch.cuda.empty_cache()

        # 如果最后一个YES段还未结束，记录它
        if current_segment_start is not None:
            yes_segments.append((current_segment_start, lp * per_frame))
        
        resp = "没有出现该行为"
        if len(yes_segments):
            resp = ""
            # 输出YES段的时间区间
            for i,(start, end) in enumerate(yes_segments):
                m1,s1=start//60,start%60
                m2,s2=end//60,end%60
                if i == 0:
                    resp += f"区间{m1}分{s1}秒到{m2}分{s2}秒"
                else:
                    resp += f",区间{m1}分{s1}秒到{m2}分{s2}秒"
            resp += f"检测出目标行为"
        f = open("./progress.txt","w+")
        f.write(str(1))
        f.close()        
        return yes_segments


    def predict(self, prompt, video_data, threshold,skipframe):   

        video_data,vlength = encode_video(video_data)
        # question = "请判断是否出现写字教学的情况"
        prompt2model = f""" 
        ###下面有2类任务：
        type1:统计类任务，你需要根据问题统计数量
        type2:行为判断任务，你需要判断是否出现该行为
        你需要判断问题属于哪一类任务。      
        只需要按下面的格式输出，不需要输出其它内容：
        {{
        'type': 1 or 2
        }}
        ###下面开始
        问题：{prompt}
        回答：
        """
        tmp = self.model_predict(prompt2model, [], 0.8)
        print(tmp)
        task = extract_res(tmp)
        torch.cuda.empty_cache()
        if task == 1:
            resp = "统计类任务尚未开发"
        elif task == 2:
            resp = self.vllm(prompt,video_data,vlength,threshold,skipframe)
        else:
           resp = "请重新描述"
        return resp