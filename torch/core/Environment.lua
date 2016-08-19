-- require 'rltorch'
 
 --- Describe a sequential environment with one sensor and one feedback
local Environment = torch.class('rltorch.Environment'); 
 
--- Initialize the environment (with parameters if needed)
function Environment:__init(world,sensor,feedback)
  self.world=world
  self.sensor=sensor
  self.feedback=feedback 
  self.action_space=world.action_space
  self.observation_space=sensor.observation_space
end
 
---Update the environment given the chosen action
-- @params agent_action the action of the agent
-- @returns observation,feedback,finished
function Environment:step(agent_action)
  self.world:step(agent_action)
  local obs=self.sensor:observe( self.world)
  local feed=self.feedback:feedback( self.world)
  local finished=self.feedback:finished( self.world)
  return {obs,feed,finished}
end

-- Reset the environment
function Environment:reset(parameters)
  self.world:reset(parameters)
  local obs=self.sensor:observe( self.world)
  return obs
end 

--- Close the environment (at the end of the process)
function Environment:close()
  self.world:close()
end
