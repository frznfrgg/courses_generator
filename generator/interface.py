import random
import uuid
from typing import List

from moviepy.editor import AudioFileClip, VideoFileClip, concatenate_videoclips
from mutagen.mp3 import MP3


class GeneratorInterface:
    def __init__(self, video_paths: List[str], text_path: str):
        self.video_paths = video_paths
        self.text_path = text_path

    def generate_mp3(self):
        pass

    def generate_mp4(self):
        # load generated audio and videoclips, extrafct durations
        audio = MP3(self.audio_path)
        target_duration = audio.info.length
        all_clips = [VideoFileClip(p) for p in self.video_paths]

        # resample videoclips to form a resulting video
        clips = []
        accum_duration = 0

        while accum_duration < target_duration:
            random.shuffle(all_clips)
            for clip in all_clips:
                remaining = target_duration - accum_duration

                if clip.duration <= remaining:
                    clips.append(clip)
                    accum_duration += clip.duration
                else:
                    clips.append(clip.subclip(0, remaining))  # trim length if needed
                    accum_duration += remaining
                    break

        # save results results
        final_clip = concatenate_videoclips(clips, method="compose")
        audio_clip = AudioFileClip(self.audio_path)
        final_clip = final_clip.set_audio(audio_clip)
        result_path = "gen_results/" + str(uuid.uuid4()) + ".mp4"
        final_clip.write_videofile(result_path)
    
    def generate(self):
        self.generate_mp3()
        self.generate_mp4()


