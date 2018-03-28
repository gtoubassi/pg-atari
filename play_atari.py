#!/usr/bin/env python
#
import sys
import numpy as np
import os
import random
import time
import argparse
import pg_network
from atari_environment import AtariEnvironment
from state import State

parser = argparse.ArgumentParser()
parser.add_argument("--screen-capture-freq", type=int, default=250, help="record screens for a game this often")
parser.add_argument("--normalize-weights", action='store_true', default=True, help="if set weights/biases are normalized like torch, with std scaled by fan in to the node")
parser.add_argument("--save-model-freq", type=int, default=10000, help="save the model once per 10000 training sessions")
parser.add_argument("--observation-steps", type=int, default=50000, help="train only after this many stesp (=4 frames)")
parser.add_argument("--model", help="tensorflow model checkpoint file to initialize from")
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
    
    while not environment.isGameOver():
  
        if state is None:
            action = random.randrange(environment.getNumActions())
        else:
            screens = np.reshape(state.getScreens(), (1, 84, 84, 4))
            action = random.randrange(environment.getNumActions())
            #XXX action = pgn.inference(screens)

        # Make the move
        oldState = state
        reward, state, isTerminal = environment.step(action)
        
        if time.time() - lastLogTime > 60:
            print('  ...frame %d' % environment.getEpisodeFrameNumber())
            lastLogTime = time.time()

        if isTerminal:
            state = None

    episodeTime = time.time() - startTime
    print('Episode %d ended with score: %d (%d frames in %fs for %d fps)' %
        (environment.getGameNumber(), environment.getGameScore(),
        environment.getEpisodeFrameNumber(), episodeTime, environment.getEpisodeFrameNumber() / episodeTime))
    environment.resetGame()

    return environment.getGameScore()


while True:
    score = playGame()
