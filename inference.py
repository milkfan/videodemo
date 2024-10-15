import io
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
import argparse
import torch
from PIL import Image
from decord import VideoReader, cpu    # pip install decord
import re
import time
import pickle as pkl

MODEL_PATH = "openbmb/MiniCPM-V-2_6-int4"
LOCAL_FLAG = True # control whether to update model from HuggingFace
PROGRESS_FILE = "./progress.txt"

# #最大帧数
MAX_NUM_FRAMES = 3600 # 最大支持分析1小时视频
STEP = 1 # 检测精度1秒
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

parser = argparse.ArgumentParser(description="CogVLM2-Video CLI Demo")
parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=4)
args = parser.parse_args([])


def encode_video(video_data):
    '''
    视频采样，1秒提取1帧

    '''
    # bridge.set_bridge('torch')
    mp4_stream = video_data
    # print(mp4_stream)
    # f=open(f"/vl/{int(time.time())}.pkl","wb+")
    # pkl.dump(video_data,f)
    vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    print('FPS:', sample_fps)
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = frame_idx[:MAX_NUM_FRAMES] # 只处理前MAX_NUM_FRAMES帧
    vlength =  len(frame_idx)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print(f'num frames: {len(frames)}, vlength: {vlength}')
    return frames, vlength


class llm():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=LOCAL_FLAG
            # padding_side="left"
        )        
        # Load the model
        if args.quant == 4:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                local_files_only=LOCAL_FLAG,
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
                local_files_only=LOCAL_FLAG,
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
                local_files_only=LOCAL_FLAG,
                torch_dtype=TORCH_TYPE,
                cache_dir="/vl/model",
                trust_remote_code=True
            ).eval().to(DEVICE)
    
    def model_predict(self, prompt, video_data, temperature):
        '''
        调用大语言模型
        '''
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

    def predict(self, prompt, video_data, threshold, skipframe):   
        '''
        根据用户的输入检测视频
        '''
        video_data, vlength = encode_video(video_data)  # 将视频处理为长度为vlength的数组
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
        res = self.model_predict(prompt2model, [], 0.8)
        torch.cuda.empty_cache()
        print(f"Task predicted: {res}")        

        def extract_res(resp):
            pattern = r"(\d+)"
            type_ids = re.findall(pattern, resp)
            return int(type_ids[0])
        task = extract_res(res)

        if task == 1:
            resp = "统计类任务尚未开发"
        elif task == 2:
            # 行为检测
            resp = self.vllm(prompt, video_data, vlength, threshold, skipframe)
        else:
           resp = "请重新描述"
        return resp
    
    def vllm(self, prompt, frames_list, vlength, threshold=5, skipframe=5):
        '''
        每隔skipframe帧检测视频中是否出现prompt中的行为，
        视频格式是长度为vlength的图像列表，
        要求检测出的行为持续threshold秒
        '''
        l = len(frames_list)
        per_frame = vlength / l  # 每帧对应的秒数，一般为1
        # print(per_frame)
        step = int(skipframe)  # 每skipframe秒检测一次
        yes_segments = []  # 存储YES段的时间区间
        current_segment_start = None  # 记录当前YES段的开始时间
        current_segment_end = None  # 记录当前YES段的开始时间
        temperature = 0.5
        lp = 0
        skipframe = int(skipframe)
       
        while lp < l:
            prompt_text = f""" 
            下面有任务你需要回答：
            task:{prompt}
            只需要按下面的格式输出，不需要输出其它内容：
            {{
            'result':"Yes or No"
            }}
            """
            res = self.model_predict(prompt_text, [frames_list[lp]], temperature)
            # 记录视频处理进度
            f = open(PROGRESS_FILE,"w+")
            f.write(str(lp/l))
            f.close()
            if "Yes" in res or "yes" in res:
                print(f"{lp}秒检测到行为!")
                # 向前回溯，找到 "Yes" 的开始
                print("向前回溯")
                if current_segment_start is None:
                    current_segment_start = lp
                    # 按精度STEP回溯检查前面的帧
                    for i in range(lp - STEP, max(lp - skipframe, 0), -STEP):
                        res_back = self.model_predict(prompt_text, [frames_list[i]], temperature)
                        if "Yes" in res_back or "yes" in res_back:
                            print(f"{i}秒检测到行为!")
                            current_segment_start = i
                        else:
                            break
                print (f"开始时间{current_segment_start}秒")

                # 向后扩展，直到找到 "No"
                # 第一步：间隔skipframe粗略地找
                print("向后回溯，粗略")                
                current_segment_end = lp + skipframe
                while current_segment_end < l:
                    res_forward = self.model_predict(prompt_text, [frames_list[current_segment_end]], temperature)
                    if "Yes" in res_forward or "yes" in res_forward:
                        print(f"{current_segment_end}秒检测到行为!")
                        current_segment_end += skipframe  # 继续向后扩展
                    else:
                        break
                lp  = current_segment_end

                # 第二步：按精度STEP精确地找
                print("向后回溯，精确")
                end_min = current_segment_end - skipframe
                if current_segment_end >= l:
                    current_segment_end = l
                tmp = end_min
                for i in range(end_min + STEP, current_segment_end, STEP):
                    res_forward = self.model_predict(prompt_text, [frames_list[i]], temperature)
                    if "Yes" in res_forward or "yes" in res_forward:
                        tmp = i 
                        print(f"{i}秒检测到行为!")
                    else:
                        break
                current_segment_end = tmp
                print (f"结束时间{current_segment_end}秒")


                # 判断视频长度是否超过了最小时长
                if (current_segment_end - current_segment_start) * per_frame >= threshold:
                    # 记录YES段的结束时间
                    yes_segments.append((current_segment_start * per_frame, current_segment_end * per_frame))
                else:
                    print(f"片段时长不满足条件")

                current_segment_start = None  # 重置片段开始时间
                current_segment_end = None  # 重置片段结束时间

            lp += skipframe  # 按每skipframe秒步进            
            torch.cuda.empty_cache()
        
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
            resp += f"检测出目标行为\n"
        print(resp)

        # 输出100%进度
        f = open("./progress.txt","w+")
        f.write(str(1))
        f.close()       

        return yes_segments


    