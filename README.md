
This package is a Reinforcement Learning package written in LUA for Torch. It main features are (for now):
* Different environments are provided, from classical RL environments, ATARI games, to special ones like the `multiclass classification environment` that casts a classification learning problem to a RL problem.  
  * Classic reward-based policies: Policy gradient, recurrent policy gradient, approximated q-learning with experience replay (also known as deep Q learning)
  * Imitation-based policies: Stochastic gradient-based imitation policy
  * Predictive policies: policies which goal is to predict an output and not to maximize a reward
* The different policies can be easily used with `openAI Gym` directly in python by using the `lutorpy` package

# News
* 19th of August 2016: Major update ! 
  * Now, environments have been splitted in three components: world, sensor and task. It allows one to easily specify different sensors, and different problems on the same world. It also greatly increase the readability of the platform
  * Tutorials have been updated
  * WARNING: Adaptation to openAI GYm will be done in the next few days...

# Dependencies

Lua: 
* [Torch7](http://torch.ch/docs/getting-started.html#_)
* nn, dpnn
* logroll, json, alewrap, sys, paths, tds
```bash
luarocks install nn
luarocks install dpnn
luarocks install logroll
luarocks install json
luarocks install sys
luarocks install paths
luarocks install tds
git clone https://github.com/deepmind/xitari.git && cd xitari && luarocks make && cd .. && rm -rf xitari
git clone https://github.com/deepmind/alewrap.git && cd alewrap && luarocks make && cd .. && rm -rf alewrap
```

For using openAI Gym:
* openai gym
* lutorpy

# Installation

* `cd torch && luarocks make`
* Install [lutorpy](https://github.com/imodpasteur/lutorpy) and [OpenAI Gym](https://gym.openai.com/)
* lauch the python script (example.py)

# Documentation

The package if composed of these different elements:
* [Core](doc/core.md): the core classes
* [Sensors](doc/sensors.md): the different generic sensors (that are not specific to a particular world)
* [Policies](doc/policies.md): different (learning) policies
* [Environments](doc/environments.md): different environments
  * [Classic Control Tasks](doc/env_classiccontrol.md): Classic control tasks
  * [Atari](doc/env_atari.md): Atari environments
  * [Classic Machine Learning](doc/env_classicmachinelearning.md): We also provide some environments that correspond to classical machine learning problems seen as RL environments 
* [Tools](doc/tools.md): different tools

# OpenAI Gym

(To update) The interface with the open AI Gym package is explained [Here](doc/openai.md)

# Tutorials

The tutorials are avalaible here: [Tutorials](doc/tutorials.md)

# FAQ

1. When installing Lutorpy, Luajit is not being detected.

Check that pkg-config can find luajit. The following should return at least one result:

```
pkg-config --list-all | grep luajit
```

If there are no results, then your `.pc` file for luajit is probably not in the right place. Try something like the following:

```
ln -s /path/to/torch/exe/luajit-rocks/luajit-2.0/etc/luajit.pc /usr/local/lib/pkgconfig/luajit.pc
```

2. Exception related to no display, such as `pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"`.

OpenAI Gym needs some sort of display to record results. On ubuntu, you may try to install xvfb/asciinema. Then try running example.py like so:

```
xvfb-run -s "-screen 0 1400x900x24" python example.py
```

Author: Ludovic DENOYER -- The code is provided as if, some bugs may exist.....
