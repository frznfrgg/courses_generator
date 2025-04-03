import os
import platform
import subprocess

import cv2
import numpy as np
import torch

from Wav2Lip import audio
from Wav2Lip.face_detection.api import FaceAlignment, LandmarksType
from Wav2Lip.models.wav2lip import Wav2Lip


class Wav2LipInterface:
    def __init__(
        self,
        video_path,
        audio_path,
        output_path="results/result_voice.mp4",
    ):
        self.video_path = video_path
        self.audio_path = audio_path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_path = os.path.join(base_dir, output_path)
        self.temp_dir = os.path.join(base_dir, "temp")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_size = 96
        self.checkpoint_path = os.path.join(base_dir, "checkpoints/wav2lip_gan.pth")
        self.pads = [0, 10, 0, 0]
        self.nosmooth = False
        self.box = [-1, -1, -1, -1]
        self.wav2lip_batch_size = 1
        self.fps = 25
        self.resize_factor = 1
        self.crop = [0, -1, 0, -1]
        self.rotate = False

    def process_video(self):
        video_stream = cv2.VideoCapture(self.video_path)

        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break

            y1, y2, x1, x2 = self.crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)
        return full_frames

    def process_audio(self):
        mel_step_size = 16
        temp_path = os.path.join(self.temp_dir, "temp.wav")
        if not self.audio_path.endswith(".wav"):
            command = "ffmpeg -y -i {} -strict -2 {}".format(self.audio_path, temp_path)

            subprocess.call(command, shell=True)
            audio_path = temp_path

        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                "Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again"
            )

        mel_chunks = []
        mel_idx_multiplier = 80.0 / self.fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1
        return mel_chunks

    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T :]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self, images):
        detector = FaceAlignment(
            LandmarksType._2D, flip_input=False, device=self.device
        )

        batch_size = 1

        while 1:
            predictions = []
            try:
                for i in range(0, len(images), batch_size):
                    predictions.extend(
                        detector.get_detections_for_batch(
                            np.array(images[i : i + batch_size])
                        )
                    )
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError(
                        "Image too big to run face detection on GPU. Please use the --resize_factor argument"
                    )
                batch_size //= 2
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.pads
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite(
                    os.path.join(self.temp_dir, "faulty_frame.jpg"),
                    image,
                )  # check this frame where the face was not detected.
                raise ValueError(
                    "Face not detected! Ensure the video contains a face in all the frames."
                )

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.nosmooth:
            boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [
            [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
            for image, (x1, y1, x2, y2) in zip(images, boxes)
        ]

        del detector
        return results

    def datagen(self, frames, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            face_det_results = self.face_detect(frames)
        else:
            y1, y2, x1, x2 = self.box
            face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.img_size, self.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size // 2 :] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
                mel_batch = np.reshape(
                    mel_batch,
                    [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1],
                )

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    def _load(self, checkpoint_path):
        if "cuda" in self.device:
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )
        return checkpoint

    def load_model(self, path):
        model = Wav2Lip()
        checkpoint = self._load(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace("module.", "")] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()

    def generate(self):
        mel_chunks = self.process_audio()
        full_frames = self.process_video()
        full_frames = full_frames[: len(mel_chunks)]

        gen = self.datagen(full_frames.copy(), mel_chunks)
        for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
            if i == 0:
                model = self.load_model(self.checkpoint_path)

                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter(
                    os.path.join(self.temp_dir, "result.avi"),
                    cv2.VideoWriter_fourcc(*"DIVX"),
                    self.fps,
                    (frame_w, frame_h),
                )
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(
                self.device
            )
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(
                self.device
            )

            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)
        out.release()
        command = "ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}".format(
            self.audio_path,
            os.path.join(self.temp_dir, "result.avi"),
            self.output_path,
        )
        subprocess.call(command, shell=platform.system() != "Windows")
