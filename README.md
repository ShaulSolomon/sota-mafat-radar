# sota-mafat-radar

MAFAT Radar Challenge - Can you distinguish between humans and animals in radar tracks?

> The participants’ goal is to classify segments of radar tracks of humans or animals using the I/Q signal matrix as an input. The task at hand is a binary classification task; the tracked objects are either humans or animals. The data is real-world data, gathered from diverse geographical locations, different times, sensors, and qualities (high- and low-signal to noise ratio—SNR).

Created a streaming pipeline to collect the data, add shifts, and process into either spectograns and scalograms.
The data was then fed into several CNN Models and a TCN (Temporal Convolutional Network) Model.
The output of the models was combined using Bagging to produce our test predictions.

**Final Place: 24th out of 47 (229 Initially)**

---

We initially was running the models within notebooks and then shifted to scripts and wandb.
As such, some of the code is still preserved with the old format, and some is using the latest scripts.<br>

The most up-to-date script that runs the models is:
sota-mafat-radar/src/scripts/alexnet-pytorch-3d.py

