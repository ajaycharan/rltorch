
# MountainCar

The moutain car environment
* `MountainCarWorld`: the world
* Sensors:
  * `MountainCar_CompleteSensor`: computes a 1x2 tensor `{{position,speed}}`
  * `MountainCar_QtSensor`: does not provide anu output, but draws the world state in a Qt window (for visualization)
* `MountainCar_Task0`: the same task than the one described in openAIGym:
  * reward is always -1
  * the episode stops when the car reaches the goal position (i.e 0.5)

# CartPole

The cart pole environment
* `CartPoleWorld`: the world
* Sensors:
  * `CartPole_CompleteSensor`: computes a 1x4 tensor `{{x,speed_x,angle,speed_angle}}`
  * `CartPole_QtSensor`: does not provide anu output, but draws the world state in a Qt window (for visualization)
* `CartPole_Task0`: the same task than the one described in openAIGym:
 
