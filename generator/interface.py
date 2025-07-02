from typing import List
from moviepy import VideoFileClip
import xtts_ru.xtts_inference
import os
class GeneratorInterface:
    def __init__(self, video_paths: List[str], text_path: str):
        self.video_path = video_paths
        self.text_path = text_path

    def _resample_videos(self):
        pass
    
    def _convert_vid2aud(self):
        video = VideoFileClip(self.video_path)
        wav_path = self.video_path[:self.video_path.rfind(".")] + ".wav"
        video.audio.write_audiofile(wav_path)
        return wav_path
    
    def generate_mp3(self):
        wav_path = self._convert_vid2aud()
        with open(self.text_path) as f:
            text = f.readlines()
        clean_text = "".join(text).replace("/n", "")

        xtts = xtts_ru.xtts_inference.XttsInference()

        transc, audio = xtts(clean_text, wav_path)

        self.audio_path = audio
        os.remove(wav_path)
    def generate_mp4(self):
        pass
