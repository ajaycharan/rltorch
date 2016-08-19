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