 # Core Classes

The rltorch package is based on a few main classes.
* `World`: It describes a world on which agent(s) can act. 
* `Sensor`: It describes a sensor i.e a (partial) view of the world state
* `Feedback`: It provides feedback to a learning agent and describes a learning task
* `Policy`: It corresponds a policy (or agent) interacting with a world
* `Space`: This class describes spaces (for example discrete space, grid space, etc...) and is basically used to describe the action and observation spaces

# World

The class is an abstract class used to implement new environments. An environment must contain the variable:
* `action_space`: the space of the actions
* `state`: the current state of the world

It is composed of the following methods:
* `__init(parameters)` : Initialization of the world. `parameters` is a table that may contain some parameters (e.g size of the world). 
* `step(agent_action)`: It applies the `agent_action` to the environment and modifies the world state. 
* `observation reset()` : It resets the environment by sampling a new initial state.
*  `close()` : to close the world 

# Sensor
A sensor computes a partial view of a world:
* `_init(world)`: initialize a new sensor on a particular world
* `observe(world)`: computes the partial view (e.g a vector, a matrix, a picture, ...)


# Feedback
A `Feedback` object describes a learning problem: 
* `__init(world)`: initialize the feedback on a particular world
* `feedback(world)`: provides a feedback (w.r.t to the last action applied on the world). The feedback can be of any type (reward, supervised information, label,...)
* `finished(world)`: Tells if the episode is finished or note

# Policy

This class describes an agent actiong in an environment. It is based on a `Sensor` (see below). It correponds to `P(a_t | sensor(o_t))` where `sensor(o_t)` is the observation of the environment throught the sensor. 

The methods are:
* `__init(observation_space,action_space,sensor)`: It initializes the policy (given a particular sensor if needed)
* `new_episode(initial_observation,informations)`: must be called at the begining of a trajectory (just after the Environment:reset() function). Note that `informations` can be used to give external information to the policy.
* `observe(observation)`: must be called before sampling a new action
* `sample()`: It samples an action given the last observed observation.
* `feedback(feedback)`: It provides feedback (can be a scalar or any other structure depending on the nature of the policy) corresponding to the last sampled action
* `end_episode(feedback)`: must be called at the end of a trajectory. `feedback` corresponds to the feedback provided for the whole trajectory (e.g the total reward when using the policy gradient algorithm)


