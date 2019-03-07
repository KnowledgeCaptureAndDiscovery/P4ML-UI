'''
    spider.tests: test_featurization_base.py

    Max Morrison

    Unit tests for the spider.featurization.audio module
'''

import librosa
import numpy as np
import os
import unittest
import warnings

from spider.featurization.audio_featurization import AudioFeaturization

class TestAudio(unittest.TestCase):

    def setUp(self):
        """Setup the audio testing environment for Python unittest"""

        data_path = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(data_path, 'data/32318.mp3')

        self._audio, self._sr = librosa.load(data_file, sr=None)
        self._featurizer = AudioFeaturization(self._sr, 0.050, 0.025)

    def test_single_time_series(self):
        """Verify the output shape of a single audio time series"""

        features = self._featurizer.produce([self._audio])

        # Different versions of librosa / ffmpeg may load audio
        # with small differences in sample length
        self.assertTrue(features[0].shape == (307, 34) or
                        features[0].shape == (308, 34))

    def test_multiple_time_series(self):
        """Verify the output shape of a list of audio time series"""

        data = np.array_split(self._audio, 5)

        features = self._featurizer.produce(data)

        for feature in features:
            self.assertEqual(feature.shape, (60, 34))

    def test_insufficient_length_time_series(self):
        """Verify short time series provide warning"""

        x = np.arange(self._sr * 0.050)
        short_series = np.sin(2 * np.pi * x / self._sr)

        data = [np.array([]), short_series[:-1], short_series, self._audio]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            features = self._featurizer.produce(data)

            self.assertEqual(len(w), 2)
            self.assertEqual(features[0].size, 0)
            self.assertEqual(features[1].size, 0)
            self.assertEqual(features[2].shape, (1, 34))
            self.assertTrue(features[3].shape == (307, 34) or
                            features[3].shape == (308, 34))

    def test_multi_channel_time_series(self):
        """Verify that mono and stereo audio channels are correctly handled"""

        mono = self._audio
        stereo = np.stack((mono, mono))
        quad = np.stack((mono, mono, mono, mono))

        features = self._featurizer.produce([mono, stereo])

        self.assertTrue(features[0].shape == (307, 34) or
                        features[0].shape == (308, 34))
        self.assertTrue(features[1].shape == (307, 34) or
                        features[1].shape == (308, 34))

        with self.assertRaises(ValueError):
            self._featurizer.produce([quad])
