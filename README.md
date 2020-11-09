# sota-mafat-radar

[MAFAT Radar Challenge - Can you distinguish between humans and animals in radar tracks?](https://competitions.codalab.org/competitions/25389)

> The participants’ goal is to classify segments of radar tracks of humans or animals using the I/Q signal matrix as an input. The task at hand is a binary classification task; the tracked objects are either humans or animals. The data is real-world data, gathered from diverse geographical locations, different times, sensors, and qualities (high- and low-signal to noise ratio—SNR).

Created a streaming pipeline to collect the data, add shifts, and process into either spectograms or scalograms as input to models.
The data was then fed into several CNN Models and a TCN (Temporal Convolutional Network) Model.
For the private phase, the output of the two top-performing models was combined using Bagging to produce our final private test predictions.

**Final Results:**
- **Private Test: 0.7935 ROC AUC - 24th place out of 47 teams**
- **Public Test: 0.8031 ROC AUC - 78th place out of 228 teams**

---

We were initially running the models within notebooks, and at a later stage shifted to using scripts and logging in Weights & Biases.
As such, some of the code is still preserved with the old format, and some is using the latest scripts.<br>

To see experiment results, go to the [W&B project page](https://wandb.ai/sota-mafat/sota-mafat-base/?workspace=user-sota-mafat)

The most up-to-date script that runs the models is:  sota-mafat-radar/src/scripts/alexnet-pytorch-3d.py

