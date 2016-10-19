require 'nn'
require 'dpnn'  
  
  --- A policy based on REINFORCE for discrete action spaces

local PolicyGradient = torch.class('rltorch.PolicyGradient','rltorch.Policy'); 

--- the policy_module is a R^n -> nb_actions >sampling vector
--- ARGUMENTS= 
----- policy_module = the policy module (takes a 1*n matrix to a 1*n vector with one for the chosen action using dpnn)
----- max_trajectory_size  = the maximum length of the trajectories
----- optim = the optim method (e.g optim.adam)
----- optim_params = the optim initial state 
----- arguments.size_memory_for_bias = number of steps to aggregate for computing the bias in policy gradient -- the n last reward values are used to correct the reward obtained.
function PolicyGradient:__init(observation_space,action_space,arguments)
  rltorch.Policy.__init(self,observation_space,action_space)   
    
  assert(arguments.policy_module~=nil)
  assert(arguments.max_trajectory_size~=nil)
  assert(arguments.optim~=nil)
  assert(arguments.optim_params~=nil)
    
  
  self.optim=arguments.optim
  self.optim_params=arguments.optim_params
  
  self.policy_module=arguments.policy_module
  self.max_trajectory_size=arguments.max_trajectory_size
  
  assert(arguments.size_memory_for_bias>=0)
  self.memory=rltorch.ScalarMemory(arguments.size_memory_for_bias)
  
  
  self.models_utils=rltorch.ModelsUtils()
  self:init()
end

function PolicyGradient:init()    
  self.params, self.grad = self.models_utils:combine_all_parameters(self.policy_module) 
  self.modules=self.models_utils:clone_many_times(self.policy_module,self.max_trajectory_size)
  self.delta=torch.Tensor(1,self.action_space.n):fill(1)
  
  self.feval = function(params_new)
    if self.params ~= params_new then
        self.params:copy(params_new)
    end
    
    self.grad:zero()
    
    ---- Calcul du discounted reward à chaque pas de temps
    self.memory:push(self.reward_trajectory)
    local mean=self.memory:mean()
    if (mean==nil) then mean=0 end
    
    for t=1,#self.trajectory.actions do
      local out=self.modules[t].output
      self.modules[t]:reinforce(torch.Tensor({self.reward_trajectory-mean}))
      self.modules[t]:backward(self.trajectory.observations[t],self.delta)
    end
    return -self.reward_trajectory,self.grad           
  end
end

function PolicyGradient:new_episode(initial_observation,informations)
  self.trajectory=rltorch.Trajectory()
  self.last_sensor=self.models_utils:deepcopy(initial_observation)
  self.trajectory:push_observation(self.last_sensor)
    
end

function  PolicyGradient:observe(observation)  
  self.last_sensor=self.models_utils:deepcopy(observation)

  self.trajectory:push_observation(self.last_sensor)  
end

function PolicyGradient:feedback(reward)
  self.trajectory:push_feedback(reward)  
end

function PolicyGradient:sample()
  local out=self.modules[self.trajectory:get_number_of_observations()]:forward(self.last_sensor)
  local vmax,imax=out:max(2)
  self.trajectory:push_action(imax[1][1])
  return(imax[1][1])
end

function PolicyGradient:end_episode(feedback)
  self.reward_trajectory=feedback
  if (self.train) then   local _,fs=self.optim(self.feval,self.params,self.optim_params)  end
end

function PolicyGradient:reset(stdv)
  self.params:copy(torch.randn(self.params:size())*stdv)
end
 

