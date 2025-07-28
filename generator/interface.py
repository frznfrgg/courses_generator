import os
import random
import subprocess
from typing import List
import re
import uuid
from moviepy.editor import AudioFileClip, VideoFileClip, concatenate_videoclips

from Wav2Lip.interface import Wav2LipInterface

from F5TTS.f5_tts.api import F5TTS
from ruaccent import RUAccent

class GeneratorInterface:
    def __init__(self, video_paths: List[str], text_path: str, final_video: str = "final_video"):
        self.video_paths = video_paths
        self.text_path = text_path
        self.final_video = final_video

    def _convert_vid2aud(self):
        video = VideoFileClip(self.video_paths[0])
        wav_path = self.video_paths[0][: self.video_paths[0].rfind(".")] + ".wav"
        video.audio.write_audiofile(wav_path)
        return wav_path

    def generate_mp3_chunks(self):
        wav_path = self._convert_vid2aud()
        with open(self.text_path) as f:
            text = f.readlines()
        clean_text = "".join(text).replace("/n", "")
        
        splitted_text = re.split(r'(?<=[.!?])\s+', clean_text)
        chunks = []
        max_length = 2200
        chunk = ""
        
        for i in splitted_text:
            i = i.strip()
            if (len(chunk) + len(i)) < max_length:
                chunk += " " + i
            else:
                chunks.append(chunk)
                chunk = i
        chunks.append(chunk)

        f5tts = F5TTS(ckpt_file="F5TTS/ckpts/model_last_inference.safetensors", vocab_file="F5TTS/ckpts/vocab.txt", device="cuda")
        accentizer = RUAccent()
        accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)
        
        aud_names = []

        for text in chunks:
            output_file_path = str(uuid.uuid4()) + ".wav"
            wav, sr, spec = f5tts.infer(
                ref_file=wav_path,
                ref_text="",
                gen_text=accentizer.process_all(text),
                file_wave=output_file_path,
                seed=None,
                remove_silence=True,
                )
            
            aud_names.append(output_file_path)

        return aud_names

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
                f.write(f"file 'Wav2Lip/{filepath}'\n")
        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            file_with_videpaths,
            "-c",
            "copy",
            f"{self.final_video}.mp4",
        ]
        subprocess.run(cmd, check=True)
    
    def _clean(self, mp3_paths: List[str], mp4_paths: List[str]) -> None:
        for path in mp3_paths:
            os.remove(path)
        for path in mp4_paths:
            os.remove(path)
        with open("list_of_videos.txt") as f:
            file_paths = f.readlines()
            for row in file_paths:
                path = row.split()[1][1:-1]  # extracting 
                os.remove(path)
        os.remove("list_of_videos.txt")
        os.remove("Wav2Lip/temp/result.avi")

    def generate_with_wav2lip(self):
        mp3_paths = self.generate_mp3_chunks()
        mp4_paths = self.create_matching_videos(mp3_paths, self.video_paths)
        output_paths = []
        for i, media_paths in enumerate(zip(mp3_paths, mp4_paths)):
            mp3_path, mp4_path = media_paths
            output_path = f"results/result_voice{i}.mp4"
            wav2lip = Wav2LipInterface(
                video_path=mp4_path, audio_path=mp3_path, output_path=output_path
            )
            wav2lip.generate()
            output_paths.append(output_path)
        self.ffmpeg_concat(output_paths)

        self._clean(mp3_paths, mp4_paths)

        return f"{self.final_video}.mp4"
