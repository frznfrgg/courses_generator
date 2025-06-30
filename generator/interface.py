from typing import List


class GeneratorInterface:
    def __init__(self, video_paths: List[str], text_path: str):
        self.video_path = video_paths
        self.text_path = text_path

    def _resample_videos(self):
        pass

    def generate_mp3(self):
        pass

    def generate_mp4(self):
        pass
