import os
import random
import uuid
from typing import List

from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips
from mutagen.mp3 import MP3

import xtts_ru.xtts_inference
from Wav2Lip.interface import Wav2LipInterface


class GeneratorInterface:
    def __init__(self, video_paths: List[str], text_path: str):
        self.video_paths = video_paths
        self.text_path = text_path

    def _convert_vid2aud(self):
        video = VideoFileClip(self.video_paths[0])
        wav_path = self.video_paths[0][: self.video_paths[0].rfind(".")] + ".wav"
        video.audio.write_audiofile(wav_path)
        return wav_path

    def generate_raw_mp3(self):
        wav_path = self._convert_vid2aud()
        with open(self.text_path) as f:
            text = f.readlines()
        clean_text = "".join(text).replace("/n", "")

        xtts = xtts_ru.xtts_inference.XttsInference()
        transc, audio = xtts(clean_text, wav_path)

        self.mp3_path = audio
        os.remove(wav_path)

    def generate_raw_mp4(self):
        # load generated audio and videoclips, extrafct durations
        audio = MP3(self.mp3_path)
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
                    clips.append(clip.subclipped(0, remaining))  # trim length if needed
                    accum_duration += remaining
                    break

        # save results results
        final_clip = concatenate_videoclips(clips, method="compose")
        self.mp4_path = str(uuid.uuid4()) + ".mp4"
        final_clip.write_videofile(self.mp4_path)

    def generate_with_wav2lip(self):
        self.generate_raw_mp3()
        self.generate_raw_mp4()
        wav2lip = Wav2LipInterface(video_path=self.mp4_path, audio_path=self.mp3_path)
        wav2lip.generate()
