import numpy as np
from scipy.ndimage.interpolation import rotate
from code.utils import experiment_utils as utils


def rotate_iq(iq, theta):
    """
    Rotate IQ matrix.

    Arguments:
    iq -- {ndarray} -- 'iq_sweep_burst' array
    theta -- {float} degrees to rotate by, in radians

    Returns:
    Rotated iq matrix
    """
    M = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    iq_ = np.array((iq.real, iq.imag))
    iq_ = iq_.transpose(1, 0, 2)
    t_iq = M @ iq_
    return t_iq[:, 0, :] + 1j*t_iq[:, 1, :]


def rotate_spectogram(data, segment_id, angle: int):
    """
        Rotates a segment spectogram.

        Arguments:
        data -- {dictionary} -- python dictionary of python numpy arrays
        segment_id -- {int} -- the segment_id number
        angle -- {int} degrees to rotate by

        Returns:
        Rotated spectogram
        """
    segment_index = np.where(data['segment_id'] == segment_id)
    segment = data['iq_sweep_burst'][segment_index]
    segment = segment.reshape(segment.shape[1], -1)
    spectogram = utils.calculate_spectrogram(segment)
    return rotate(spectogram, angle=angle)


def rotate_track(data, track_id, angle: int):
    """
    Concatenates segments for a track and rotates the combined spectograms.

    Arguments:
    data -- {dictionary} -- python dictionary of python numpy arrays
    track_id -- {int} -- the track_id number of the wanted segments
    angle -- {int} degrees to rotate by

    Returns:
    Rotated spectogram
    """
    track, doppler = utils.concatenate_track(data, track_id)
    assert utils.get_label(data, track_id) is not None, "No definitive label, can't rotate track"
    spectogram = utils.calculate_spectrogram(track)
    return rotate(spectogram, angle=angle)

def resplit_track_random(track, start=0, n_splits=5):
    """
    Randomly splits a track into n new segments.

    Arguments:
    track -- {ndarray} -- spectogram created from IQ matrix
    start -- {int} -- the starting point of range of possible starting indices for new segments (Default=0)
    n_splits -- {int} number of new segments to create (Default=5)

    Returns:
    Dictionary of new segments like {track_index: segment_array}
    """
    assert track.shape[1] > 32, "Track is too short to re-split"
    indices = np.random.choice(range(start, track.shape[1]-32), n_splits)
    vert_indices = np.random.choice(range(track.shape[1]-128),n_splits)
    segments = {}
    for j, i in enumerate(indices):
        vi = vert_indices[j]
        segment = track[vi:vi+128, i: i+32].copy()
        segments[i] = segment
    return segments


def split_rotation(data, track_id, angle: int, n_splits=5):
    """
    Rotates track spectograms and creates new random segments.

    Arguments:
    data -- {dictionary} -- python dictionary of python numpy arrays
    track_id -- {int} -- the track_id number of the wanted segments
    angle -- {int} degrees to rotate by
    n_splits -- {int} number of new segments to create (Default=5)

    Returns:
    Dictionary of new rotated segments like {track_index: segment_array}
    """
    rot_track = rotate_track(data, track_id, angle)
    assert rot_track.shape[1] >= 96, "Track is too short to re-split"
    start = rot_track.shape[1]//6
    return resplit_track_random(rot_track, start=start, n_splits=n_splits)
    


def vertical_flip(iq):
    return iq.real + -1j*iq.imag


def horizontal_flip(iq):
    return iq.real * -1 + 1j*iq.imag