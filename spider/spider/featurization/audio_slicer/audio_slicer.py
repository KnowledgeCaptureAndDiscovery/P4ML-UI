import numpy as np
import stopit
import sys
from spider.featurization_base import FeaturizationTransformerPrimitiveBase

Inputs = list
Outputs = list

class AudioSlicer(FeaturizationTransformerPrimitiveBase[Inputs, Outputs]):

    def __init__(
        self,
        sampling_rate=44100,
        frame_length=3.0,
        overlap=0.0,
        pad=True
    ):
        """
        DARPA D3M Audio Slicer Primitive
        
        Arguments:
            - sampling_rate: integer-valued uniform sampling rate of the
                incoming audio data
            - frame_length: float-valued duration in seconds that defines the
                length of the sliced clips
            - overlap: float-valued duration in seconds that defines the step
                size taken along the time series during adjacent clip
                extraction
            - stretch: boolean value indicating whether clips shorter than
                frame_length should be padded with zeros to a duration of
                exactly frame_length
        """
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.overlap = overlap
        self.pad = pad
        self.samples_per_clip = int(frame_length * sampling_rate)
        self.step = max(int((frame_length - overlap) * sampling_rate), 1)

        if overlap >= frame_length:
            raise ValueError(
                'AudioSlicer: Consecutive audio frames of length ' +\
                str(frame_length) + ' seconds cannot facilitate ' +\
                str(overlap) + 'seconds of overlap.'
            )


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

                # Iterate through audio extracting clips
                clips = []
                for i in xrange(0, len(datum), self.step):
                    if i + self.samples_per_clip <= len(datum):
                        clips.append(datum[i : i + self.samples_per_clip])
                    elif self.pad:
                        clips.append(
                            np.concatenate([
                                datum[i:],
                                np.zeros(
                                    self.samples_per_clip - len(datum[i:]))
                            ])
                        )

                features.append(np.array(clips))

        if timer.state == timer.EXECUTED:
            return features
        else:
            raise TimeoutError('AudioSlicer exceeded time limit')
            