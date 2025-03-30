import cv2
import librosa
import numpy as np


class LipSync:
    def __init__(self, video_path: str, audio_path: str, face_detector_path: str):
        self.face_detector_path = face_detector_path
        self.audio_path = audio_path
        self.video_path = video_path
        

    def get_result(self):
        frames = self.extract_frames(self.video_path)
        faces = []
        for frame in frames:
            face = self.detect_face(frame)
            faces.append(face)
        

    def extract_frames(self) -> np.array:
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames)

    def detect_face(
        self,
        image: np.array,
    ) -> np.array:
        detector = cv2.FaceDetectorYN.create(
            model=self.face_detector_path,
            config="",
            input_size=(image.shape[0], image.shape[1]),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000,
        )

        height, width = image.shape[:2]

        detector.setInputSize((width, height))
        faces = detector.detect(image)

        x, y, w, h = map(int, faces[1][0][:4])
        cropped_image = image[y : y + h, x : x + w, :]
        return cropped_image

    def audio_to_mel(self, target_sr=16000) -> np.array:
        audio, sr = librosa.load(self.audio_path, sr=None)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        n_fft = 800
        hop_length = 640
        n_mels = 80
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=target_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram
