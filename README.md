# Implementation of A3C (Asynchronous Advantage Actor-Critic)

This is a tensorflow implementation of Asynchronous advantage actor-critic algorithm for CNN-LSTM as function approximator

## Original Paper
[here](https://arxiv.org/abs/1602.01783)

## Demo

![Breakout_v0](/src/Breakout_v0.gif)

## Results

Training on Breakout-v0 is done with Nvidia GeForce GTX 1070 GPU for 28 hours

## Total Scores Vs Number of iteration (Breakout_v0) 

![Scores](/src/Training_Breakout_Total_Scores.png)

## Episode Length Vs Number of iteration (Breakout_v0)

![Episode_Length](/src/Training_Breakout_episode_length.png)



## Dependencies

* python 3.5
* tensorflow 1.1.0
* opencv 3.2.0
* openAI


## Usage

For Training Run:

```
$ python3 trainer.py
```

For Demo Run:

```
$ python3 play.py
```

## Credit

Got important help form this [project](https://github.com/MatheusMRFM/A3C-LSTM-with-Tensorflow)



# A3C_CNN_LSTM
