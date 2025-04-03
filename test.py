# from Wav2Lip.interface import Wav2LipInterface
import Wav2Lip.interface

video_path = "data/subclip.mp4"
audio_path = "data/audio_subclip.mp3"

wav2lip = Wav2Lip.interface.Wav2LipInterface(
    video_path=video_path, audio_path=audio_path
)

wav2lip.generate()