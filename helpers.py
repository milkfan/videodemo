from moviepy.editor import VideoFileClip, concatenate_videoclips

# 处理视频并生成片段
def process_video(video_path, time_segments):
    # 打开视频文件
    video = VideoFileClip(video_path)

    # 生成片段
    segments = []
    for segment_start,segment_end in time_segments:
        # 格式化时间戳
        start_timestamp = f"{int(segment_start // 60)}分{int(segment_start % 60)}秒"
        end_timestamp = f"{int(segment_end // 60)}分{int(segment_end % 60)}秒"

        # 创建视频片段
        segment_end = min(segment_end,video.duration)
        segment = video.subclip(segment_start, segment_end)
        segment_path = f'./outputs/{start_timestamp}-{end_timestamp}.mp4'
        segment.write_videofile(segment_path, codec='libx264', audio=False)
        segments.append(segment_path)

    return segments

def load_video_data(video_path):
    with open(video_path, 'rb') as file:
        video_data = file.read()
    return video_data