import gradio as gr
import requests
import os
import time
from helpers import process_video, load_video_data

URL = 'http://localhost:5000'
POST_INTERVAL = 2

# 大模型AI Agent
class ChatAgent:
    def __init__(self):
        self.url= URL

    def answer(self, video_path, prompt, max_new_tokens, threshold, skipframe):
        url = self.url + '/video_qa'
        print(video_path)
        files = {'video': open(video_path, 'rb')}
        data = {'question': prompt, 'threshold': threshold,'skipframe':skipframe}
        response = requests.post(url, files=files, data=data)
        if response.status_code != 200:
            return f"Something went wrong: {response.text}"
        else:
            return response.json()["answer"]

agent = ChatAgent()


# 交互函数
def upload_video(gr_video):
    '''
    点击上传按钮
    '''
    if gr_video is None:
        gr.Warning("请先选择本地视频，或打开摄像头录制视频!", duration=3)
        return ( None, 
            gr.update(value="上传视频", interactive=True),
            gr.update(value="开始检测", interactive=False)
        )
    else:
        print(f"Get video: {gr_video}")
        return (
            gr.update(interactive=True),
            gr.update(value="已上传", interactive=False),
            gr.update(value="开始检测", interactive=True)
        )

def gradio_answer(video_path, option, threshold, skipframe):
    '''
    视频检测
    '''
    if len(option) == 0 or video_path is None:
        return option, gr.update(interactive=True), gr.update(interactive=True)
    response = agent.answer(video_path=video_path, prompt=option, max_new_tokens=200, threshold=threshold, skipframe=skipframe)
    print(f"Question: {option} Answer: {response}")

    # 根据大模型返回的结果，生成视频切片
    seg = process_video(video_path, response)
    print(f"返回数量为{len(seg)}的片段")
    return seg, gr.update(interactive=True), gr.update(interactive=True)

def gradio_reset():
    '''
    点击清空按钮
    '''
    return (
        None,None,
        gr.update(value=None, interactive=True),
        gr.update(value="上传视频", interactive=True),
        gr.update(value="开始检测", interactive=False)
    )

        
def process_data():
    '''
    视频处理进度跟踪
    '''
    bar = gr.Progress(track_tqdm=True)
    progress = 0
    bar(progress)
    time.sleep(POST_INTERVAL)
    while progress < 0.9999:
        # 间歇
        time.sleep(POST_INTERVAL)
        try:
            url = URL + '/video_progress'
            response = requests.get(url)
            if response.status_code != 200:
                print(response.json()["error"])
            else:
                res = response.json()["progress"]
                print(res)
                progress = float(res)   
        except:
            pass
        bar(progress)
    return "当前进度: 100%，需等待视频剪辑"

def clear_answer():
    '''
    清空上一次检测结果，用于连续检测
    '''
    return None, gr.update(interactive=False), gr.update(interactive=False)


# 页面布局
with gr.Blocks(title="视频检测项目案例", css="#files {height: 150px} #chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as my_demo:
    with gr.Row(): 
        # 左侧上传视频         
        with gr.Column(scale=2, visible=True) as video_upload:
            # 只有mp4格式才能支持实时的视频捕获
            up_video = gr.Video(interactive=True, include_audio=True, elem_id="video_upload", height=400, format="mp4", label="视频") 
            upload_button = gr.Button(value="上传视频", interactive=True, variant="primary")


        # 右侧检测视频    
        with gr.Column(scale=3, visible=True) as video_detect:
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    threshold = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=3,
                        step=1,
                        interactive=True,
                        label="视频片段最小长度（秒）",
                    )

                with gr.Column(scale=1, min_width=300):
                    skipframe = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        interactive=True,
                        label="检测间隔（秒/次）",
                    )

            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    option = gr.Dropdown(choices=["检测是否使用手机", "检测是否坐下"], label="选择检测类型", value="检测是否使用手机", interactive=True)
                with gr.Column(scale=1, min_width=300):
                    run_button = gr.Button("💭开始检测", interactive=False)
                    clear_button = gr.Button("🔄清空")
            
            progress_output = gr.Textbox(label="视频检测进度", interactive=False)
            output_videos = gr.Files(label="视频检测结果", interactive=False, elem_id="files")


        # 左侧组件交互
        # 获取上传/录制的视频保存的地址
        upload_button.click(upload_video, [up_video], [up_video, upload_button, run_button]) 

        # 右侧组件交互
        # 清空上一次检测结果
        run_button.click(clear_answer, [], [output_videos, run_button, clear_button]) 
        # 开始检测    
        run_button.click(gradio_answer, [up_video,option,threshold,skipframe], [output_videos, run_button, clear_button])
        # 跟踪检测进度
        run_button.click(fn=process_data, inputs=[], outputs = progress_output)
        # 重置状态
        clear_button.click(gradio_reset, [], [output_videos, progress_output, up_video, upload_button, run_button], queue=False)  

    gr.Markdown("#### 检测出的视频片段")
    with gr.Row():    
        @gr.render(inputs=output_videos)
        def display_videos(videos):
            if videos and len(videos) != 0:
                num = len(videos)
                left = (3 - num % 3) % 3
                for video in videos:
                    print(video)
                    with gr.Column(scale=1, min_width=300):
                        gr.PlayableVideo(value=video, label=f"{video.split('/')[-1]}", height=200)
                for i in range(left):
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("")

if __name__ == '__main__':
    my_demo.launch(share=True,server_name="0.0.0.0", server_port=7868)
