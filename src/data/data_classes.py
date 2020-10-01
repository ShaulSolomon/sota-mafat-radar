from typing import Dict, List, TypedDict
import numpy as np


class Segment(object):
    def __init__(self, track_id: int = None, segment_id: int = None, iq_matrix: np.ndarray = None,
                 doppler_burst: np.ndarray = None, geolocation_id: int = None, geolocation_type: str = None,
                 sensor_id: int = None, snr_type: str = None, date_index: int = None, target_type: str = None,
                 is_validation: bool = False):
        self.track_id = track_id
        self.segment_id = segment_id
        self.iq_matrix = iq_matrix
        self.doppler_burst = doppler_burst
        self.geolocation_id = geolocation_id
        self.geolocation_type = geolocation_type
        self.sensor_id = sensor_id
        self.snr_type = snr_type
        self.date_index = date_index
        self.target_type = target_type
        self.is_validation = is_validation

    # TODO function for turning IQ matrix into spectogram and scalogram, with option of adding doppler burst
    # TODO function to flip segment-as-IQ matrix horizontally
    # TODO function to flip segment-as-IQ matrix vertically
    # TODO function to flip segment-as-spectogram horizontally
    # TODO function to flip segment-as-spectogram vertically
    # TODO function to flip segment-as-scalogram horizontally
    # TODO function to flip segment-as-scalogram vertically


class track(object):
    def __init__(self, segments: List[Segment]):
        self.segments = {}
        segments.sort(key=lambda x: x.segment_id)

    # TODO how to ensure that one track object contains only segments from that track?
    # TODO create multiple tracks if segments are from different tracks?
    # TODO function to split track into sub-tracks according to the is_validation attribute of sequential segments.
    # TODO function to create new segments from spectograms
    # TODO function to create new segments from scalograms

