# Predictive Policy

# Predictive Recurrent Policy Gradient Tutorial

We explain here how the `PredictiveRecurrentPolicyGradient` policy can be used for making prediction (seen as a sequential process). It aims at targetting a muticlass classification problem where the sequential process acquires one feature at each timestep as described  in `Gabriel Dulac-Arnold, Ludovic Denoyer, Philippe Preux, Patrick Gallinari: Sequential approaches for learning datum-wise sparse representations. Machine Learning 89(1-2): 87-122 (2012)`. At the end of the acquisition process, the policy has to predict the category of the given example. 

First, we load a dataset and create the corresponding environment

```lua

require ('optim') 
require('rltorch')
require('svm')

math.randomseed(os.time())


local TEST_SIZE=100 -- The number of trajectories
local TRAIN_SIZE=100 -- The number of trajectories
local NB_ITERATIONS=1000 -- The number of training example to sample for each trajectory
local NB_FEATURES=5

local PROPORTION_TRAIN=0.5
local data,labels,nb_categories = unpack(rltorch.RLFile():read_libsvm('datasets/breast-cancer_scale'))


--- Load libsvm
local parameters={}
parameters.training_examples, parameters.training_labels,parameters.testing_examples,parameters.testing_labels = unpack(rltorch.RLFile():split_train_test(data,labels,PROPORTION_TRAIN))

--- Create the world
world = rltorch.FeaturesAcquisitionClassificationWorld(parameters)
sensor=rltorch.FeaturesAcquisitionClassification_Sensor(world)
task=rltorch.FeaturesAcquisitionClassification_Task(world)

local size_input=sensor.observation_space:size()[2]
local nb_actions=world.action_space.n
print("Size input = "..size_input)
print("Nb_actions = "..nb_actions)
print("Nb categories = "..nb_categories)

-- Creating the predictive policy
local N=10 -- the size of the latent space
local STDV=0.01
local initial_state=torch.Tensor(1,N):fill(0)
-- Creating the policy module which maps the latent space to the action space.
local module_policy=nn.Sequential():add(nn.Linear(N,nb_actions)):add(nn.SoftMax()):add(nn.ReinforceCategorical()); module_policy:reset(STDV)
local initial_recurrent_module = rltorch.RNN():rnn_cell(size_input,N,N); initial_recurrent_module:reset(STDV)
local recurrent_modules={}
for a=1,nb_actions do
  recurrent_modules[a]=rltorch.GRU():gru_cell(size_input,N)
  recurrent_modules[a]:reset(STDV)
end
--the module that maps the latent space to the final prediction
local predictive_module=nn.Linear(N,nb_categories); predictive_module:reset(STDV)

-- the predictive criterion
local criterion=nn.CrossEntropyCriterion()

local arguments={
    policy_module = module_policy,
    predictive_module=predictive_module,
    initial_state = initial_state,
    N = N,
    initial_recurrent_module = initial_recurrent_module,
    recurrent_modules = recurrent_modules,
    max_trajectory_size = NB_FEATURES+1,
    optim=optim.adam,
    optim_params= {
        learningRate =  0.001  
      },
    size_memory_for_bias=100,
    criterion=criterion
  }
  
policy=rltorch.PredictiveRecurrentPolicyGradient(sensor.observation_space,world.action_space,arguments)
local train_losses={}
local test_losses={}
for i=1,NB_ITERATIONS do    
    --- Evaluation on the test set
    policy.train=false
    local total_loss_test=0
      for j=1,TEST_SIZE do
      world:reset(true)
      policy:new_episode(sensor:observe(world))      
      local feedback
      for t=1,NB_FEATURES do
        local action=policy:sample()      
        local observation=sensor:observe(world)        
        feedback=task:feedback(world); assert(feedback.target~=nil)
        local done=task:finished(world)
        policy:observe(observation)
      end
      local prediction=policy:predict()
      local loss_test=criterion:forward(prediction,feedback.target)
      total_loss_test=total_loss_test+loss_test      
    end
    total_loss_test=total_loss_test/TEST_SIZE
    test_losses[i]=total_loss_test
    
    -- Evaluation + training on training examples
    policy.train=true
    local total_loss_train=0
    for j=1,TRAIN_SIZE do
      world:reset(false)
      policy:new_episode(sensor:observe(world))
      local feedback
      for t=1,NB_FEATURES do
        local action=policy:sample()      
        local observation=sensor:observe(world)        
        feedback=task:feedback(world); assert(feedback.target~=nil) -- the feedback must contain a 'target' value (which is the value to predict at the end of the epiosde)
        local done=task:finished(world)
        policy:observe(observation)
      end
      local prediction=policy:predict()
      local loss_train=criterion:forward(prediction,feedback.target)
      total_loss_train=total_loss_train+loss_train 
      policy:end_episode(feedback.target)
    end  
    total_loss_train=total_loss_train/TRAIN_SIZE
    train_losses[i]=total_loss_train
  
    if (i%1==0) then gnuplot.plot({"Training Loss",torch.Tensor(train_losses),"~"},{"Testing Loss",torch.Tensor(test_losses),"~"}) end
end
env:close()
```
