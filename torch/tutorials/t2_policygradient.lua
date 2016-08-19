

require ('optim') 
require('rltorch')

MAX_LENGTH=1000
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

-- Creating the policy module (neural network)
local module_policy=nn.Sequential():add(nn.Linear(size_input,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical())
module_policy:reset(0.01)

local arguments={
    policy_module = module_policy,
    max_trajectory_size = MAX_LENGTH,
    optim=optim.adam,
    optim_params= {
        learningRate =  0.01  
      },
    size_memory_for_bias=100
  }

-- create the policy
policy=rltorch.PolicyGradient(sensor.observation_space,world.action_space,arguments)

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
