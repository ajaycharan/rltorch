
# Atari_v0

The Atari environment with 18 actions

* initialization parameters:
  * `parameters.rom` = the rom filename
* action space: `Discrete(18)` - 18 possible actions 
  * ACTION_MEANING = {  0 : "NOOP",    1 : "FIRE",    2 : "UP",    3 : "RIGHT",    4 : "LEFT",    5 : "DOWN",    6 : "UPRIGHT",    7 : "UPLEFT",    8 : "DOWNRIGHT",    9 : "DOWNLEFT",    10 : "UPFIRE",    11 : "RIGHTFIRE",    12 : "LEFTFIRE",    13 : "DOWNFIRE",    14 : "UPRIGHTFIRE",    15 : "UPLEFTFIRE",    16 : "DOWNRIGHTFIRE",    17 : "DOWNLEFTFIRE",}
* observation space: a 3D tensor of size (3,210,160) where min value is 0 and max value is 255. Each value corresponds to one pixel (r,g,b)
* reward: the reward returned by ALE (i.e score increase/decrease)
* reset: start a new game from the beginning
* render
  * `mode=qt,fps=(optional)`

# Atari_Breakout_v0

The same as `Atari_v0` but actions are limited to actions 2,4,5 (for the breakout game)
* action space: `Discrete(3)`

 
