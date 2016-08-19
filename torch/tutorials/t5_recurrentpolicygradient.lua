

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


