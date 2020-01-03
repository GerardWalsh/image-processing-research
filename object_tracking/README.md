# Generic, single object tracking
Tracking of a single, generic object in a RGB input stream

First paper
--------------------------------------------
:heavy_check_mark: [An In-Depth Analysis of Visual Tracking with Siamese Neural Networks] [[Paper]](https://arxiv.org/pdf/1707.00569.pdf)

## Realtime object trackers
The trackers listed here should perform inference, as defined by VOT benchmark, at rate faster than which the input is fed into the algorithm (~20-30FPS)
### Siamese network trackers

:heavy_check_mark: [Fully-Convolutional Siamese Networks for Object Tracking] [[Paper]](https://arxiv.org/abs/1606.09549)[[Code]](https://github.com/bertinetto/siamese-fc)

## Highly accurate trackers (not real-time)
The trackers listed here do not perform (mostly) at real-time inference rates, but generally around single digits (1-5 FPS)  
:heavy_check_mark: [Good Features to Correlate for Visual Tracking] [[Paper]](https://arxiv.org/abs/1704.06326)[[Code]](https://github.com/egundogdu/CFCF)

# Benchmark

:heavy_check_mark: [Visual Object Tracking challenge] [[Paper 2013]](http://www.votchallenge.net/vot2013/Download/vot_2013_paper.pdf)[[Paper 2018]](http://prints.vicos.si/publications/365)[[Toolkit]](https://github.com/votchallenge/vot-toolkit)
