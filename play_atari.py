#!/usr/bin/env python
#
import sys
import numpy as np
import os
import random
import time
import argparse
import pg_network
import sys
from atari_environment import AtariEnvironment
from state import State

parser = argparse.ArgumentParser()
parser.add_argument("--screen-capture-freq", type=int, default=250, help="record screens for a game this often")
parser.add_argument("--normalize-weights", action='store_true', default=True, help="if set weights/biases are normalized like torch, with std scaled by fan in to the node")
parser.add_argument("--save-model-freq", type=int, default=10000, help="save the model once per 10000 training sessions")
parser.add_argument("--observation-steps", type=int, default=50000, help="train only after this many stesp (=4 frames)")
parser.add_argument("--model", help="tensorflow model checkpoint file to initialize from")
parser.add_argument("--games-per-epoch", type=int, default=100, help="Number of games to play per training epoch (default 100)")
parser.add_argument("--learning-rate", type=float, default=.001, help="Learning rate (default .001)")
parser.add_argument("--training-passes-per-epoch", type=int, default=1, help="How many passes over training data to make per epoch (default 1)")
parser.add_argument("--use-rms-prop", action='store_true', default=False, help="Use the RMSPropOptimizer instead of Adam")
parser.add_argument("rom", help="rom file to run")

args = parser.parse_args()

print('Arguments: %s' % (args))

baseOutputDir = 'game-out-' + time.strftime("%Y-%m-%d-%H-%M-%S")
os.makedirs(baseOutputDir)

environment = AtariEnvironment(args, baseOutputDir)

pgn = pg_network.PolicyGradientNetwork(environment.getNumActions(), baseOutputDir, args)

def playGame():

    startTime = lastLogTime = time.time()
    stateReward = 0
    state = None
    xs = []
    ys = []

    while not environment.isGameOver():

        if state is None:
            action = random.randrange(environment.getNumActions())
        else:
            x = state.getScreens()
            action = pgn.inference(np.reshape(x, (1, 84, 84, 4)))
            xs.append(x)
            ys.append(action)

        # Make the move
        oldState = state
        reward, state, isTerminal = environment.step(action)

        if time.time() - lastLogTime > 60:
            print('  ...frame %d' % environment.getEpisodeFrameNumber())
            lastLogTime = time.time()

    episodeTime = time.time() - startTime
    print('Episode %d ended with score: %d (%d frames in %fs for %d fps)' %
        (environment.getGameNumber(), environment.getGameScore(),
        environment.getEpisodeFrameNumber(), episodeTime, environment.getEpisodeFrameNumber() / episodeTime))
    gameScore = environment.getGameScore()
    environment.resetGame()

    return gameScore, xs, ys

def trainEpoch():
  
    games = []
    for i in range(args.games_per_epoch):
        games.append(playGame())

    scores, all_xs, all_ys = zip(*games)
    cutoff = np.percentile(scores, 70)

    training_data = []
    for score, xs, ys in zip(scores, all_xs, all_ys):
        if score >= cutoff:
            for x, y in zip(xs, ys):
                training_data.append((x, y))
    
    batch_size = 20
    
    for training_pass in range(args.training_passes_per_epoch):
      random.shuffle(training_data)
      batches = [training_data[x:x+batch_size] for x in range(0, len(training_data), batch_size)]
    
      print("Training pass %d of %d with %d batches..." % (training_pass+1, args.training_passes_per_epoch, len(batches)), end='')
      sys.stdout.flush()
      total_loss = 0
      for i, batch in enumerate(batches):
          x, y = zip(*batch)
          total_loss += pgn.train(x, y)
      average_loss = total_loss / len(batches)
      print(" Done (ave loss: %f)" % average_loss)
    
while True:
   trainEpoch()
