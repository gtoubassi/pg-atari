#!/usr/bin/env python
#
import sys
import numpy as np
import os
import random
import replay
import time
import argparse
import dqn
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

def runEpoch(minEpochSteps):
    stepStart = environment.getStepNumber()
    startGameNumber = environment.getGameNumber()
    epochTotalScore = 0

    while environment.getStepNumber() - stepStart < minEpochSteps:
    
        startTime = lastLogTime = time.time()
        stateReward = 0
        state = None
        
        while not environment.isGameOver():
      
            if state is None:
                action = random.randrange(environment.getNumActions())
            else:
                screens = np.reshape(state.getScreens(), (1, 84, 84, 4))
                action = dqn.inference(screens)

            # Make the move
            oldState = state
            reward, state, isTerminal = environment.step(action)
            
            # Record experience in replay memory and train
            if oldState is not None:
                clippedReward = min(1, max(-1, reward))
                replayMemory.addSample(replay.Sample(oldState, action, clippedReward, state, isTerminal))

                if environment.getStepNumber() > args.observation_steps and environment.getEpisodeStepNumber() % 4 == 0:
                    batch = replayMemory.drawBatch(32)
                    dqn.train(batch, environment.getStepNumber())
        
            if time.time() - lastLogTime > 60:
                print('  ...frame %d' % environment.getEpisodeFrameNumber())
                lastLogTime = time.time()

            if isTerminal:
                state = None

        episodeTime = time.time() - startTime
        print('%s %d ended with score: %d (%d frames in %fs for %d fps)' %
            ('Episode' if isTraining else 'Eval', environment.getGameNumber(), environment.getGameScore(),
            environment.getEpisodeFrameNumber(), episodeTime, environment.getEpisodeFrameNumber() / episodeTime))
        epochTotalScore += environment.getGameScore()
        environment.resetGame()
    
    # return the average score
    return epochTotalScore / (environment.getGameNumber() - startGameNumber)


while True:
    aveScore = runEpoch(args.train_epoch_steps) #train
    print('Average training score: %d' % (aveScore))
    aveScore = runEpoch(args.eval_epoch_steps, evalWithEpsilon=.05) #eval
    print('Average eval score: %d' % (aveScore))
