 
 --- The gaol of a sensor is to a world state to an observation
 local Sensor = torch.class('rltorch.Sensor'); 

function Sensor:__init(world)
  self.observation_space=nil
end

function Sensor:observe(world)
  assert(false,"No process function")
end

