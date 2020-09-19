#!/usr/bin/env python3

import random
import pandas as pd
from src.models import arch_setup
from torch.utils.data import DataLoader

def make_dummy_dataset(tracks_amount, track_legths):
    df = {'track_id': [], 'segment_id': []}
    x = 0
    for track_id, track_length in enumerate(track_legths):
        for _ in range(track_length):
            df['track_id'].append(track_id)
            df['segment_id'].append(x)
            x += 1
    # TODO aggregate dataset into one track per row, use concatenate tracks
    return pd.DataFrame(data=df)


def test_dataset_working():
    tracks_amount = random.randint(10, 20)
    track_legths = [random.randint(100, 200) for _ in range(tracks_amount)]
    dataset = make_dummy_dataset(tracks_amount, track_legths)

    cfg = {'get_shifts': False,
         'block_size' : 50}

    ds = arch_setup.DS2(df=dataset, labels=dataset, config=cfg)
    loader = DataLoader(dataset=ds, batch_size=1, shuffle=False, num_workers=0)
    count = sum([1 for sample in loader])
    assert count == sum(track_legths)


def test_dataset_with_shifts():
    track_legths = [100, 50, 40]
    tracks_amount = len(track_legths)

    dataset = make_dummy_dataset(tracks_amount, track_legths)

    segment_size = 32 # under assumption of this segment_size
    for l in track_legths:
        assert l > segment_size
    expected_sample_count = sum([l - segment_size + 1 for l in track_legths])

    for block_size in [30, 100]:
        cfg={'get_shifts': True,  # this option assumes DS2 creates all the possible shifts in each track
             'block_size': block_size}

        ds = arch_setup.DS2(df=dataset, labels=dataset, config=cfg)
        loader = DataLoader(dataset=ds, batch_size=1, shuffle=False, num_workers=0)
        count = sum([1 for sample in loader])
        assert count == expected_sample_count

test_dataset_working()