import numpy as np
import stopit
import sys
import warnings
from .utils import audio_feature_extraction
from spider.featurization_base import FeaturizationTransformerPrimitiveBase

Inputs = list
Outputs = list

class AudioFeaturization(
    FeaturizationTransformerPrimitiveBase[Inputs, Outputs]):

    def __init__(
        self,
        sampling_rate=44100,
        frame_length=0.050,
        overlap=0.025):
        """
        Darpa D3M Audio Featurization Primitive

        Arguments:
           - sampling_rate: integer-valued uniform sampling rate of the incoming
                audio data
           - frame_length: float-valued duration in seconds that defines the
                length of the audio processing window
           - overlap: float-valued duration in seconds that defines the step
                size taken along the time series during subsequent
                processing steps
        """
        self.sampling_rate = sampling_rate
        self.frame_length=frame_length
        self.overlap = overlap
        self.step = max(int((frame_length - overlap) * sampling_rate), 1)

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
                if datum.shape[0] < self.frame_length * self.sampling_rate:
                    warnings.warn(
                        'Cannot construct a frame of length ' +           \
                        str(self.frame_length) +                          \
                        ' seconds from input datum ' +                    \
                        str(i) + ' of length ' +                          \
                        str(datum.shape[0] / float(self.sampling_rate)) + \
                        ' seconds.',
                        RuntimeWarning
                    )
                    features.append(np.array([]))
                    continue

                # Perform audio feature extraction
                features.append(
                    audio_feature_extraction(
                        datum,
                        self.sampling_rate,
                        self.frame_length * self.sampling_rate,
                        self.step
                    ).T
                )

        if timer.state == timer.EXECUTED:
            return features
        else:
            raise TimeoutError('AudioFeaturization exceeded time limit')
