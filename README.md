# PG Atari

This repo implements a very simple (the simplest?) policy gradient reinforcement learning approach.  It implements REINFORCE (no baseline, no actor/critic).

Previous experimentation playing atari used Deep Q Learning and can be found here: [dqn-atari](https://github.com/gtoubassi/dqn-atari)), and maxed out at scoring 1150 on Space Invaders.  Experiments with REINFORCE got me to average performance of 1850, which is a nice improvement.  However this was achieved by only cherry picking well performing episodes for training.  So I'd play 400 episodes, pick the top 30% in terms of score, and "REINFORCE" learn on those.

Sutton and Barto's ["Reinforcement Learning: An Introduction"](http://incompleteideas.net/book/bookdraft2017nov5.pdf) covers this in Chapter 13 "Policy Gradient Methods".

A publicly viewable spreadsheet of recent experiments can be [found here](https://docs.google.com/spreadsheets/d/18V_EJ3mrgAwHSs7pakgrMJiUPZ5HntzZX32SHgIDxLA/edit?usp=sharing).
