#!/usr/bin/env python3

import random
import pandas as pd
import numpy as np
from src.models import arch_setup
from src.data.get_data_pipeline import get_track_level_data
from torch.utils.data import DataLoader


def make_dummy_dataset(track_lengths):
    columns = ['geolocation_type', 'geolocation_id', 'sensor_id', 'snr_type', 'date_index', 'target_type']
    df = {'track_id': [], 'segment_id': [], 'iq_sweep_burst': [], 'doppler_burst': [], 'geolocation_id': [],
          'sensor_id': [], 'geolocation_type': [], 'snr_type': [], 'date_index': [], 'target_type': [],
          'is_validation': []}
    x = 0
    for track_id, track_length in enumerate(track_lengths):
        for i in range(track_length):
            df['track_id'].append(track_id)
            df['segment_id'].append(x)
            df['iq_sweep_burst'].append(np.random.random((128, 32)) + np.random.random((128, 32)) * 1j)
            df['doppler_burst'].append(np.random.random(32))
            df['is_validation'].append(False)
            # df['is_validation'].append(True) if i % 20 == 0 else df['is_validation'].append(False)
            for col in columns:
                df[col].append(1)
                # df[col].append(random.randint(0, 1)) if i % 10 == 0 else df[col].append(1)
            x += 1
    return df


def test_dataset_working():
    tracks_amount = random.randint(10, 20)
    track_lengths = [random.randint(100, 200) for _ in range(tracks_amount)]
    dataset = make_dummy_dataset(track_lengths)
    dataset = get_track_level_data(dataset)
    cfg = {'get_shifts': False,
         'block_size' : 50}

    ds = arch_setup.DS2(data_records=dataset, config=cfg)
    loader = DataLoader(dataset=ds, batch_size=1, shuffle=False, num_workers=0)
    sample_counts = [sample['labels'].shape[1] for sample in loader]
    count = sum(sample_counts)
    print(f'Total segments generated: {count}')
    expected_length = sum(track_lengths)
    print(f'Total segments expected: {expected_length}')
    print(f'Segments missing: {expected_length - count}')
    print(f'Tracks created: {tracks_amount}')
    assert (expected_length - count) == tracks_amount
    assert count + tracks_amount == expected_length


def test_dataset_with_shifts():
    track_lengths = [100, 50, 40]
    tracks_amount = len(track_lengths)
    dataset = make_dummy_dataset(track_lengths)
    dataset = get_track_level_data(dataset)
    segment_size = 32 # under assumption of this segment_size
    for l in track_lengths:
        assert l > segment_size
    expected_samples = [l for l in track_lengths]
    expected_sample_count = sum(expected_samples)
    print(f'Tracks created: {tracks_amount}')
    print(f'Total segments expected: {expected_sample_count}')
    for block_size in [30, 100]:
        cfg = {'get_shifts': True,  # this option assumes DS2 creates all the possible shifts in each track
             'block_size': block_size}

        ds = arch_setup.DS2(data_records=dataset, config=cfg)
        loader = DataLoader(dataset=ds, batch_size=1, shuffle=False, num_workers=0)
        samples = []
        count = 0
        for sample in loader:
            count += sample['segments'].shape[1]
            samples.append(sample)
        print(f'Total segments generated: {count}')
        print(f'Segments missing: {expected_sample_count - count}')
        assert count + tracks_amount == expected_sample_count

test_dataset_with_shifts()