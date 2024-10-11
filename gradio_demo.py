import gradio as gr
import requests
import os
import time
from helpers import process_video, load_video_data



class ChatAgent:
    def __init__(self):
        pass

    def answer(self, video_path, prompt, max_new_tokens, threshold,skipframe):
        url = 'http://127.0.0.1:5000/video_qa'
        print(video_path)
        files = {'video': open(video_path, 'rb')}
        data = {'question': prompt, 'threshold': threshold,'skipframe':skipframe}
        response = requests.post(url, files=files, data=data)
        if response.status_code != 200:
            return f"Something went wrong: {response.text}"
        else:
            return response.json()["answer"]


def gradio_reset():
    return (
        None,None,
        gr.update(value=None, interactive=True),
        # gr.update(placeholder='Please upload your video first', interactive=False),
        gr.update(value="Upload & Start", interactive=True)
    )


def upload_video(gr_video):
    if gr_video is None:
        return None, gr.update(interactive=True, placeholder='Please upload video/image first!'), gr.update(
            interactive=True)
    else:
        print(f"Get video: {gr_video}")
        return (
            gr.update(interactive=True),
            # gr.update(interactive=True, placeholder='Type and press Enter'),
            gr.update(value="Start", interactive=False)
        )


def gradio_ask(user_message, chatbot):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot


def gradio_answer(video_path, option, threshold, skipframe):
    if len(option) == 0 or video_path is None:
        return option
    response = agent.answer(video_path=video_path, prompt=option, max_new_tokens=200, threshold=threshold,skipframe=skipframe)
    print(f"Question: {option} Answer: {response}")
    seg = process_video(video_path,response)
    return seg

agent = ChatAgent()

def main():
    with gr.Blocks(title="VideoHub",
                   css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:
        with gr.Row():            
            with gr.Column(scale=0.5, visible=True) as video_upload:
                with gr.Tab("视频", elem_id='video_tab'):
                    up_video = gr.Video(interactive=True, include_audio=True, elem_id="video_upload", height=360, format="mp4")

                upload_button = gr.Button(value="上传视频并检测", interactive=True, variant="primary")
                threshold = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=1,
                    step=1,
                    interactive=True,
                    label="阈值（秒）",
                )
                skipframe = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=1,
                    step=1,
                    interactive=True,
                    label="检测频率（秒/次）",
                )
            bar = gr.Progress(track_tqdm=True)
            def process_data():
                f = open("./progress.txt","w+")
                f.write(str(0))
                f.close()
                progress = 0
                bar(progress)
                while progress < 0.9999:
                    time.sleep(1)
                    try:
                        f = open("./progress.txt","r")
                        progress = float(f.read())
                        f.close()
                    except:
                        pass
                    bar(progress)
                return "当前进度: 100%"
                
            with gr.Column(visible=True) as input_raws:
                output_videos = gr.Files(label="输出视频片段")
                progress_output = gr.Textbox(label="Progress", interactive=False)
                with gr.Row():
                    with gr.Column(scale=0.5):
                        option = gr.Dropdown(choices=["检测是否使用手机", "检测是否坐下"], label="选择检测类型", interactive=True)

                    with gr.Column(scale=0.15):
                        run = gr.Button("💭开始检测")
                    with gr.Column(scale=0.15, min_width=0):
                        clear = gr.Button("🔄清空")
            upload_button.click(upload_video, [up_video], [up_video, upload_button])     
            run.click(gradio_answer, [up_video,option,threshold,skipframe], [output_videos])
            run.click(fn=process_data, inputs=[], outputs = progress_output)
            clear.click(gradio_reset, [],[output_videos,progress_output, up_video, upload_button], queue=False)  
    
    demo.launch(share=True,server_name="0.0.0.0", server_port=7868)


if __name__ == '__main__':
    main()
