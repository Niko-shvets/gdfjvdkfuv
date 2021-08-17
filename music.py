import moviepy.editor as mp

def music_extractor(video_name:str):
        music_name = video_name.split('.')[0]
        my_clip = mp.VideoFileClip(video_name)
        my_clip.audio.write_audiofile(music_name+'.mp3')
        
        
def add_music(video:str,audio:str):
    video_name = video.split('.')[0]
    
    video = mp.VideoFileClip(video)
    audio_background = mp.AudioFileClip(audio)
        
    video = video.set_audio(audio_background)
    video.write_videofile(video_name+'.mp4')