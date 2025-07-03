import os
import random
import subprocess
from typing import List
import re

from moviepy.editor import AudioFileClip, VideoFileClip, concatenate_videoclips

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

    def generate_mp3_chunks(self) -> List[str]:
        return List[str]

    def _create_video_for_audio(
        self, audio_path: str, video_paths: List[str], output_path
    ):
        audio = AudioFileClip(audio_path)
        audio_duration = audio.duration

        clips = []
        accumulated_duration = 0

        while accumulated_duration < audio_duration:
            video_path = random.choice(video_paths)
            clip = VideoFileClip(video_path)

            remaining = audio_duration - accumulated_duration
            if clip.duration > remaining:
                clip = clip.subclip(0, remaining)

            clips.append(clip)
            accumulated_duration += clip.duration

        final_video = concatenate_videoclips(clips, method="compose")
        final_video = final_video.set_audio(audio).subclip(0, audio_duration)

        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

        final_video.close()
        audio.close()
        for c in clips:
            c.close()

    def create_matching_videos(self, audio_paths, video_paths):
        intermediate_files = []

        for idx, audio_path in enumerate(audio_paths):
            output_path = f"matched_video_{idx + 1}.mp4"
            self._create_video_for_audio(audio_path, video_paths, output_path)
            intermediate_files.append(output_path)

        return intermediate_files

    def ffmpeg_concat(self, file_list: List[str]):
        file_with_videpaths = "list_of_videos.txt"
        with open(file_with_videpaths, "w") as f:
            for filepath in file_list:
                f.write(f"file '{os.path.abspath(filepath)}'\n")
        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            file_list,
            "-c",
            "copy",
            "final_video.mp4",
        ]
        subprocess.run(cmd, check=True)

    def generate_with_wav2lip(self):
        mp3_paths = self.generate_mp3_chunks()
        mp4_paths = self.create_matching_videos(mp3_paths, self.video_paths)
        output_paths = []
        for i, mp3_path, mp4_path in enumerate(zip(mp3_paths, mp4_paths)):
            output_path = f"results/result_voice{i}.mp4"
            wav2lip = Wav2LipInterface(
                video_path=mp4_path, audio_path=mp3_path, output_path=output_path
            )
            wav2lip.generate()
            output_paths.append(output_path)
        self.ffmpeg_concat(output_path)
        return "final_video.mp4"
