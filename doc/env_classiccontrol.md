
# MountainCar_v0

The moutain car environment (same than openAI Gym). 
* Action space: `Discrete(3)`
  * Action 1 is backward, 2 is nop, and 3 is forward
* Observation space: Vector with two values
  * `v[1]`: position of the car
  * `v[2]`: speed of the car
* Reward: -1 at each step
* Finished: when the car is at the top of the hill
* Reset: random position and speed 
  * random position between -0.4 and -0.2
  * speed = 0
* Render:
  * mode=console: rendering on console
  * mode=qt,fps=(optional): display using qt

# CartPole_v0

The cart pole environment (same than openAI Gym)
* Action space: `Discrete(2)` - 1=backward, 2=forawrd
* Observation space: 4D vector : `x,speed on x, angle, speed on angle`
* Reward: 1 at each timestep
* Finished: when the pole has a position or an angle out of some threshold
  * `angle threshold = 12 * 2 * math.pi / 360`
  * `position threshold  = 2.4`
* Reset: all values randomly chosen in `[-0.05,0.05]`
* Render: 
  * `mode=console`
  * `mode=qt,fps=(optional)`
