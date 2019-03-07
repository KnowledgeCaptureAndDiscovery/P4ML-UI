import librosa
import numpy as np
import stopit
import sys
import time
import warnings
from spider.featurization_base import FeaturizationTransformerPrimitiveBase

Inputs = list
Outputs = list

class LogMelSpectrogram(
    FeaturizationTransformerPrimitiveBase[Inputs, Outputs]):

    def __init__(
        self,
        sampling_rate=44100,
        mel_bands=128,
        n_fft=1024,
        hop_length=1024
    ):
        """
        DARPA D3M Audio Spectrogram Primitive

        Arguments:
            - sampling_rate: integer-valued uniform sampling rate of the
                incoming audio data
            - mel_bands: integer number of mel frequency filters used to
                compute the spectrogram
            - n_fft: integer length in samples of the fft window
            - hop_length: integer length in samples between successive
                audio frames
        """
        self.sampling_rate = sampling_rate
        self.mel_bands = mel_bands
        self.n_fft = n_fft
        self.hop_length = hop_length

    def produce(self, inputs, timeout=None, iterations=None):
        with stopit.ThreadingTimeout(timeout) as timer:

            features = []
            for i, datum in enumerate(inputs):

                # Handle multi-channel audio data
                if datum.ndim > 2 or (datum.ndim == 2 and datum.shape[0] > 2):
                    raise ValueError(
                        'Time series datum ' + str(i) + ' found with ' + \
                        'incompatible shape ' + str(datum.shape)  + '.'
                    )
                elif datum.ndim == 2:
                    datum = datum.mean(axis=0)

                # Handle time series of insufficient length
                if datum.shape[0] < self.n_fft:
                    warnings.warn(
                        'Cannot construct a fft window of length ' +          \
                        str(self.n_fft) + ' seconds from input datum ' +      \
                        str(i) + ' of length ' + str(datum.shape[0]) + '. ' + \
                        'Returning empty np.array in output index ' +         \
                        str(i+1) + '.',
                        RuntimeWarning
                    )
                    features.append(np.array([]))
                    continue

                # Compute the mel-scaled spectrogram
                melspec = librosa.feature.melspectrogram(
                    datum,
                    sr=self.sampling_rate,
                    n_mels = self.mel_bands,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )

                # Convert spectrogram amplitude to log-scale
                features.append(librosa.power_to_db(melspec).T)

        if timer.state == timer.EXECUTED:
            return features
        else:
            raise TimeoutError('LogMelSpectrogram exceeded time limit')
            