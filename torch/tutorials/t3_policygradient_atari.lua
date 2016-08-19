require ('optim') 
require('rltorch')
require('image')
--package.path = '/home/denoyer/alewrap/?/init.lua;' .. package.path
local alewrap = require 'alewrap'


MAX_LENGTH=300  
DISCOUNT_FACTOR=1.0
NB_TRAJECTORIES=10000

--env = rltorch.MountainCar_v0()
--- Build the problem components
 
world=rltorch.AtariWorld({rom="environments/ale/roms/breakout.bin"}) -- the world
  
  task=rltorch.Atari_Task(world) -- the task
  sensor=rltorch.Atari_ImageSensor(world) -- the sensor. Here, one has to use a sensor that provides a 1xn tensor
  fsensor=rltorch.FlattenSensor(world,sensor) -- the fsensor is the flattened 1xn version of the original sensor
local size_input=fsensor.observation_space:size()[2]
print("Input size is "..size_input)
local nb_actions=world.action_space.n
print("Number of actions is "..nb_actions)

-- Creating the policy module
--  module_policy=nn.Sequential():add(nn.Linear(size_input,size_input/2)):add(nn.Tanh()):add(nn.Linear(size_input/2,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical())
module_policy=nn.Sequential():add(nn.Linear(size_input,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical())
module_policy:reset(0.001)

local arguments={
    policy_module = module_policy,
    max_trajectory_size = MAX_LENGTH,
    optim=optim.adam,
    optim_params= {
        learningRate =  0.001  
      },
    size_memory_for_bias=100
  }
  
--policy=rltorch.RandomPolicy(env.observation_space,env.action_space,sensor)
policy=rltorch.PolicyGradient(fsensor.observation_space,world.action_space,arguments)

-- Learning loop
local rewards={}
for i=1,NB_TRAJECTORIES do
    print("Starting episode "..i)
    world:reset()
    policy:new_episode(fsensor:observe(world))  
    local sum_reward=0.0
    local current_discount=1.0
    
    for t=1,MAX_LENGTH do            
      local action=policy:sample()     
      world:step(action)
      
      -- for rendering
      local render=sensor:observe(world)
      win=image.display({image=render,win=win})
      
      -- for learning
      local observation=fsensor:observe(world)
      local feedback=task:feedback(world)
      local done=task:finished(world)
      assert(feedback.reward~=nil)      
      policy:feedback(feedback.reward) -- the immediate reward is provided to the policy
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