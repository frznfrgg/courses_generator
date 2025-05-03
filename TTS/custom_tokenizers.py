import pickle
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from phonemizer import phonemize
from phonemizer.separator import Separator
from transformers import AutoProcessor, EncodecModel


class TextTokenizer:
    def __init__(self):
        with open("tokenizers_files/text_vocab.txt") as f:
            self.text_vocab = f.readline().split(", ")
        self.phon_to_idx = {phoneme: i for i, phoneme in enumerate(self.text_vocab)}
        self.idx_to_phon = {i: phoneme for i, phoneme in enumerate(self.text_vocab)}

    @staticmethod
    def extract_phonemes(text: str) -> str:
        """Convertes a text into phonemes.
        Words are separated with ' | ' and phonemes with '#'.

        Args:
            text (str): original text

        Returns:
            str: phoneme representation of a text
        """
        phonemes = phonemize(
            text,
            language="ru",
            backend="espeak",
            separator=Separator(phone="#", word=" | "),
            preserve_punctuation=False,
            with_stress=True,
            strip=True,
        )
        return phonemes

    def tokenize_text(self, text: str) -> List[int]:
        """Converts text into phonemes and tokenizes it.

        Args:
            text (str): original text

        Returns:
            List[int]:tokenized representation of a text
        """
        print(text)
        phonemes = self.extract_phonemes(text)
        tokens = [self.phon_to_idx["<s_ph>"]]
        for word in phonemes.split(" | "):
            for phoneme in word.split("#"):
                if phoneme in self.phon_to_idx.keys():
                    tokens.append(self.phon_to_idx[phoneme])
                else:
                    tokens.append(self.phon_to_idx["<UNK_ph>"])
            tokens.append(self.phon_to_idx["<space>"])
        tokens.append(self.phon_to_idx["</s_ph>"])
        return tokens

    def untokenize_text(self, tokens: List[int]) -> List[str]:
        """Converts text tokens back into phonemes

        Args:
            tokens (List[int]): tokenized representation of text

        Returns:
            List[str]: list of phonemes separated by service tokens
        """
        phonemes = []
        for token in tokens:
            if token in self.idx_to_phon.keys():
                phonemes.append(self.idx_to_phon[token])
            else:
                phonemes.append("<UNK_ph>")
        return phonemes


class AudioTokenizer:
    def __init__(self):
        with open("tokenizers_files/second_cb_to_ind.pkl", "rb") as f:
            self.second_cb_to_ind = pickle.load(f)
        with open("tokenizers_files/encodec_to_tokens.pkl", "rb") as f:
            self.encodec_to_tokens = pickle.load(f)
        with open("tokenizers_files/tokens_to_encodec.pkl", "rb") as f:
            self.tokens_to_encodec = pickle.load(f)
    
    @staticmethod
    def process_audio(path_to_audio: str) -> np.ndarray[float]:
        """Loads audio from path and converts it to numpy array.

        Args:
            path_to_audio (str): location of audio file

        Returns:
            np.ndarray: waveform representation of audio sample
        """
        waveform, orig_sr = torchaudio.load(path_to_audio)
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=24000)
        waveform = resampler(waveform)
        return waveform

    def tokenize_waveform(
        self,
        waveform: np.ndarray[float],
    ) -> Tuple[List[int], torch.tensor]:
        """Converts waveform representation of audio sample into tokens.

        Args:
            waveform (np.ndarray): waveform representation of audio

        Returns:
            Tuple[List[int], torch.tensor[[int]]]: tokenized audio and padding mask used in encodec model
        """
        model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

        inputs = processor(
            raw_audio=waveform,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_outputs = model.encode(
                inputs["input_values"], inputs["padding_mask"]
            ).audio_codes.squeeze()

        first_cb_codes = encoder_outputs[0]
        second_cb_codes = encoder_outputs[1]
        new_second_codes = torch.tensor(
            [self.second_cb_to_ind[int(code)] for code in second_cb_codes]
        )
        new_codes = torch.vstack([first_cb_codes, new_second_codes])

        tokenized = []
        for encoding in new_codes.transpose(1, 0):
            encoding = tuple([int(encoding[0]), int(encoding[1])])
            tokenized.append(self.encodec_to_tokens[encoding])
        return tokenized, inputs["padding_mask"]

    def tokens_to_audio(self, audio_tokens: List[int], padding_mask: torch.tensor):
        """Converts tokens to audio representation

        Args:
            audio_tokens (List[int]): tokenized representation of original audio
            padding_mask (torch.tensor[[int]]): padding mask used in encodec model

        Returns:
            EncodecDecoderOutput: output audio representation of encodec model
        """
        model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        encodec_tokens = []
        for token in audio_tokens:
            encodec_tokens.append(self.tokens_to_encodec[token])
        encodec_tokens = (
            torch.tensor(encodec_tokens)
            .transpose(1, 0)
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
        )
        with torch.no_grad():
            audio_values = model.decode(
                encodec_tokens,
                [None],
                padding_mask,
            )
        return audio_values
