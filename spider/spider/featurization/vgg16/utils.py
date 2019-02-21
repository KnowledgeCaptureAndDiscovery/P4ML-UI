import sys
import os.path
import urllib
import warnings
import numpy as np
from scipy import misc

def interpolate(x, shape, method):
    """ Resizes the input via various interpolations methods.
        For now just uses basic misc.imresize -- to be augmented
        Note: this function will normalize the data from 0 to 255,
        if all values equal, all will be 0

    Arguments:
        x: data to be interpolated
        shape: desired shape of the output data
        method: method to use for interpolation - 'bilinear', 
            'nearest', 'lanczos', 'bicubic', or 'cubic'

    Returns:
        y: a (224, 224, 3) np.array to be fed into the model

    Raises:
        None
    """

    y = misc.imresize(x, shape, method)
    y = y.astype(np.float32)

    return y

def download_weights(directory, filename, url):
    """ Download the weights from the given url, and save them as the given filename
        in the specified directory
        
    Arguments:
        directory: path to the directory to which the weights file should be saved
        filename: filename with which to name the downloaded weights file
        url: url from which the weights file should be downloaded

    Returns:
        filepath: the confirmed to exist path to the weights file to be
            loaded into the model

    Raises: 
        AssertError: throws an error if the given path in which to save the
            weights file does not exist
    """
    assert os.path.isdir(directory), 'path to weights directory does not exist.'
    filepath = os.path.join(directory, filename)

    if not os.path.isfile(filepath):
        print("Weights file not found, downloading from: " + str(url))
        download = urllib.FancyURLopener()
        download.retrieve(url, filepath, reporthook = dl_progress_percentage)
        print("\n")

    assert os.path.isfile(filepath), 'file was not downloaded or saved properly'

    return filepath


def dl_progress_percentage(count, blocksize, totalsize):
    """ Outputs percentage progress to stdout to track download progress
        
    Arguments:
        count: number of blocks of data retrieved
        blocksize: size of the blocks retrieved
        totalsize: total size of blocks to be retrieved

    Returns:
        None

    Raises:
        None
    """

    percent = int(count*blocksize*100/totalsize)
    sys.stdout.write("\r" + "...%d%%" % percent)
    sys.stdout.flush()
