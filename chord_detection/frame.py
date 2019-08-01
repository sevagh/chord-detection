import numpy
import math


def frame_cutter(x: numpy.ndarray, frame_size: int) -> numpy.ndarray:
    if len(x.shape) != 1:
        raise ValueError("Only 1D numpy ndarrays are supported")

    num_frames = float(x.shape[0]) / float(frame_size)
    num_frames = int(math.ceil(num_frames))
    pad = int(num_frames * frame_size - x.shape[0])
    x_pad = numpy.concatenate((x, numpy.zeros(pad)))
    for x_frame in numpy.split(x_pad, num_frames):
        yield x_frame
