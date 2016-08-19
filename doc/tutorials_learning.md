# Learning Tutorials

How can we learn a policy based on a reward ? 

# Tutorial 3 : Policy Gradient
(see tutorials/t2_policygradient)

We explain here how to use the REINFORCE algorithm on any environemnt (with discrete action space)

* Create the environment

```lua
   --- Build the problem components
  world=rltorch.CartPoleWorld() -- the world
  task=rltorch.CartPole_Task0(world) -- the task
  sensor=rltorch.CartPole_CompleteSensor(world) -- the sensor
```


* Get the size of the vectors and the number of actions
```lua
local size_input=sensor.observation_space:size()[2]
local nb_actions=world.action_space.n
```

* Build a dpnn module which will samples one action over the possible actions. The input of this module is the (1,n) vector provided by the sensor, and the output is a (1,A) onehot vector with a 1 for the chosen action. This module must implement the `reinforce` method provided in the `dpnn` package. The module used here is a linear module with a softmax and a multinomial sampling

```lua
local module_policy=nn.Sequential():add(nn.Linear(size_input,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical())
module_policy:reset(0.01) --- initialize the values of the parameters
```

* Build the policy. 
  * First, one has to decide the parameters of this policy
```lua
local arguments={
    policy_module = module_policy,
```

  * For the `PolicyGradient` class, you must provide the maximum size of the trajectories (`MAX_LENGTH` here)

```lua
    max_trajectory_size = MAX_LENGTH,
```

  * You have to provide the optimization algorithm and its parameters:
```lua
    optim=optim.adam,
    optim_params= {
        learningRate =  0.01  
      },
```

  * The average value of the last 100 trajectories will be used in reinforce to reduce the variance 
```lua 
    size_memory_for_bias=100
  }
```

* Now, we can build the policy
```lua
policy=rltorch.PolicyGradient(sensor.observation_space,world.action_space,arguments)
```

* And we can launch the learning loop. The `rewards` table is used to plot the rewards using `gnuplot`

```lua
-- Learning loop
local rewards={}
for i=1,NB_TRAJECTORIES do
    print("Starting episode "..i)
    world:reset()
    policy:new_episode(sensor:observe(world))  
    local sum_reward=0.0
    local current_discount=1.0
    
    for t=1,MAX_LENGTH do            
      local action=policy:sample()     
      world:step(action)
      local observation=sensor:observe(world)
      local feedback=task:feedback(world)
      local done=task:finished(world)
      assert(feedback.reward~=nil)  

      policy:observe(observation)      
      sum_reward=sum_reward+current_discount*feedback.reward -- computes the discounted sum of rewards
      current_discount=current_discount*DISCOUNT_FACTOR      
      if (done) then        
        break
      end
    end
    
    rewards[i]=sum_reward
    print("Reward at "..i.." is "..sum_reward)
    if (i%100==0) then gnuplot.plot(torch.Tensor(rewards),"|") end
    
    policy:end_episode(sum_reward) -- The feedback provided for the whole episode here is the discounted sum of rewards      
end

```

# Tutorial 4: Deep Q-Learning 

This is the same idea.... The only different is in the way the policy is built

```lua
require ('optim') 
require('rltorch')

MAX_LENGTH=100
DISCOUNT_FACTOR=1.0
NB_TRAJECTORIES=10000

--- Build the problem components
  world=rltorch.CartPoleWorld() -- the world
  task=rltorch.CartPole_Task0(world) -- the task
  sensor=rltorch.CartPole_CompleteSensor(world) -- the sensor.
  qsensor=rltorch.CartPole_QtSensor(world,320,200) -- (for visualization)


math.randomseed(os.time())

local size_input=sensor.observation_space:size()[2]
local nb_actions=world.action_space.n

print("Input size is "..size_input)
print("Number of actions is "..nb_actions)

-- Creating the policy module
module_policy=nn.Sequential():add(nn.Linear(size_input,size_input*2)):add(nn.Tanh()):add(nn.Linear(size_input*2,nb_actions))
--module_policy=nn.Sequential():add(nn.Linear(size_input,nb_actions)) --:add(nn.SoftMax()):add(nn.ReinforceCategorical())
module_policy:reset(0.01)

local arguments={
    policy_module = module_policy,
    max_trajectory_size = MAX_LENGTH,
    optim=optim.adam,
    optim_params= {
        learningRate =  0.01  
      },
    size_minibatch=10,
    size_memory=100,
    discount_factor=1,
    epsilon_greedy=0.1
  }
  
--policy=rltorch.RandomPolicy(env.observation_space,env.action_space,sensor)
policy=rltorch.DeepQPolicy(sensor.observation_space,world.action_space,arguments)

local rewards={}


for i=1,NB_TRAJECTORIES do
  print("Starting episode "..i)
    world:reset()
    policy:new_episode(sensor:observe(world))  
    local sum_reward=0.0
    local current_discount=1.0
    
    for t=1,MAX_LENGTH do  
      local action=policy:sample()      
      world:step(action)
      local observation=sensor:observe(world)
      --qsensor:observe(world)
      local feedback=task:feedback(world) 
      local done=task:finished(world)
      policy:feedback(feedback.reward) -- the immediate reward is provided to the policy
      policy:observe(observation)      
      sum_reward=sum_reward+current_discount*feedback.reward -- comptues the discounted sum of rewards
      current_discount=current_discount*DISCOUNT_FACTOR      
      if (done) then        
        break
      end
    end
    
    rewards[i]=sum_reward
    if (i%100==0) then gnuplot.plot(torch.Tensor(rewards),"|") end
    
    policy:end_episode() 
    if (i>10000) then policy.train=false end
end
world:close()
```

# Tutorial 5 : The recurrent policy gradient

First, as usual, you have to build the environment, sensor, etc...

```lua
require ('optim') 
require('rltorch')

MAX_LENGTH=100
DISCOUNT_FACTOR=1.0
NB_TRAJECTORIES=10000

--- Build the problem components
world=rltorch.CartPoleWorld() -- the world
task=rltorch.CartPole_Task0(world) -- the task
sensor=rltorch.CartPole_CompleteSensor(world) -- the sensor
  
math.randomseed(os.time())

-- The input size of the neural network depends on the sensor. 
local size_input=sensor.observation_space:size()[2]
local nb_actions=world.action_space.n

local N=10 -- the size of the latent space
local STDV=0.01
-- Creating the policy module which maps the latent space to the action space
local module_policy=nn.Sequential():add(nn.Linear(N,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical()); module_policy:reset(STDV)

-- the initial state in the latent space
local initial_state=torch.Tensor(1,N):fill(0)

-- This module  maps the initial state + initial observation to a new state in the latent space
local initial_recurrent_module = rltorch.RNN():rnn_cell(size_input,N,N); initial_recurrent_module:reset(STDV)

-- Now we define one module for each possible action. Each module maps the current state + new observation to a new state
local recurrent_modules={}
for a=1,nb_actions do
  recurrent_modules[a]=rltorch.GRU():gru_cell(size_input,N)
  recurrent_modules[a]:reset(STDV)
end


local arguments={
    policy_module = module_policy,
    initial_state = initial_state,
    N = N,
    initial_recurrent_module = initial_recurrent_module,
    recurrent_modules = recurrent_modules,
    max_trajectory_size = MAX_LENGTH,
    optim=optim.adam,
    optim_params= {
        learningRate =  0.001  
      },
    size_memory_for_bias=100
  }
  
policy=rltorch.RecurrentPolicyGradient(sensor.observation_space,world.action_space,arguments)

-- Learning loop
local rewards={}
for i=1,NB_TRAJECTORIES do
    print("Starting episode "..i)
    world:reset()
    policy:new_episode(sensor:observe(world))  
    local sum_reward=0.0
    local current_discount=1.0
    
    for t=1,MAX_LENGTH do            
      local action=policy:sample()     
      world:step(action)
      local observation=sensor:observe(world)
      local feedback=task:feedback(world)
      local done=task:finished(world)
      assert(feedback.reward~=nil)  
      policy:observe(observation)      
      sum_reward=sum_reward+current_discount*feedback.reward -- comptues the discounted sum of rewards
      current_discount=current_discount*DISCOUNT_FACTOR      
      if (done) then        
        break
      end
    end
    
    rewards[i]=sum_reward
    print("Reward at "..i.." is "..sum_reward)
    if (i%100==0) then gnuplot.plot(torch.Tensor(rewards),"|") end
    
    policy:end_episode(sum_reward) -- The feedback provided for the whole episode here is the discounted sum of rewards      
end
```

# Tutorial 6 : Multiclass Classification as a 0/1-reward RL problem

Here we explain how a classical multiclass classification problem can be casted to a 0/1 reward problem: 

```lua
--- This tutorial shows how a classical multiclass classification problem can be handled in a RL framework
--- In this tutorial, this is viewed as a classical RL problem

require ('optim') 
require('rltorch')
require('svm')

math.randomseed(os.time())


local NB_ITERATIONS=1000 -- The number of trajectories
local SIZE_ITERATION=1000 -- The number of training example to sample for each trajectory


--- Load the dataset from a libsvm file
local PROPORTION_TRAIN=0.5
local data,labels = unpack(rltorch.RLFile():read_libsvm('datasets/breast-cancer_scale'))
local parameters={}
parameters.training_examples, parameters.training_labels,parameters.testing_examples,parameters.testing_labels = unpack(rltorch.RLFile():split_train_test(data,labels,PROPORTION_TRAIN))

-- Create the corresponding world
world = rltorch.MulticlassClassificationWorld(parameters)
task=rltorch.MulticlassClassification_Task(world)
sensor=rltorch.MulticlassClassification_Sensor(world)
--sensor=rltorch.TilingSensor2D(env.observation_space,30,30)

local size_input=sensor.observation_space:size()[2]
local nb_actions=world.action_space.n
print("Size input = "..size_input)
print("Nb_actions = "..nb_actions)
-- Creating the policy module
local module_policy=nn.Sequential():add(nn.Linear(size_input,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical())
module_policy:reset(0.01)

local arguments={
    policy_module = module_policy,
    max_trajectory_size = SIZE_ITERATION,
    optim=optim.adam,
    optim_params= {
        learningRate =  0.01  
      },
    size_memory_for_bias=100
  }
  
policy=rltorch.PolicyGradient(sensor.observation_space,world.action_space,arguments)

local train_rewards={}
local test_rewards={}
for i=1,NB_ITERATIONS do
    
    --- Evaluation on the test set
    policy.train=false
    world:reset(true)
    policy:new_episode(sensor:observe(world)) 
    
    local sum_reward_test=0.0
    local flag=true 
    while(flag) do  
      local action=policy:sample()      
      world:step(action)
      local observation=sensor:observe(world)
      local feedback=task:feedback(world)
      local done=task:finished(world)
      assert(feedback.reward~=nil)
      sum_reward_test=sum_reward_test+feedback.reward       
      if (done) then flag=false else policy:observe(observation) end       
    end
    test_rewards[i]=sum_reward_test/parameters.testing_examples:size(1)
    print("0/1 Reward at iteration "..i.." (test) is "..sum_reward_test)  
    
    -- Evaluation + training on training examples
    policy.train=true
    world:reset(false)   
    policy:new_episode(sensor:observe(world)) 
    local sum_reward=0.0
    
    for t=1,SIZE_ITERATION do  
      local action=policy:sample()      
      world:step(action)
      local feedback=task:feedback(world)
      local done=task:finished(world)
      
      local observation=sensor:observe(world)
      assert(feedback.reward~=nil)   
      sum_reward=sum_reward+feedback.reward 
      policy:observe(observation)
      
    end
    policy:end_episode(sum_reward) -- The feedback provided for the whole episode here is the discounted sum of rewards      
    
    train_rewards[i]=sum_reward/SIZE_ITERATION
    print("0/1 Reward at iteration "..i.." (train) is "..sum_reward)  
    if (i%100==0) then gnuplot.plot({"Training accuracy",torch.Tensor(train_rewards),"~"},{"Testing accuracy",torch.Tensor(test_rewards),"~"}) end    
end
world:close()
```





 
