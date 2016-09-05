-- require 'rltorch'
 
 --- Describe a sequential world (no sensors, no feedback) 
local World = torch.class('rltorch.World'); 
 
--- Initialize the world (with parameters if needed)
--- self.action_space describes the set of possible actions
function World:__init(parameters)
  self.parameters=parameters
  self.action_space=nil
end
 
---Update the environment given the chosen action
-- @params agent_action the action of the agent
-- @returns nothing... Just update the state of the world
function World:step(agent_action)
  assert(false,"World:__init")
end

-- Reset the world 
function World:reset(parameters)
   assert(false,"World:reset")
end 

--- Close the world (at the end of the process)
function World:close()  
end

function World:clone()
  assert(false)
end

function World:toString()
  assert(false)
end
