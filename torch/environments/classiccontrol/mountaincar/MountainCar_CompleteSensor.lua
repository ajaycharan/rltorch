
local MountainCar_CompleteSensor = torch.class('rltorch.MountainCar_CompleteSensor','rltorch.Sensor'); 
    
--- returns a 2D vector with position and speed
function MountainCar_CompleteSensor:__init(world)  
  self.low = torch.Tensor({{world.min_position, -world.max_speed}})
  self.high = torch.Tensor({{world.max_position, world.max_speed}})
  self.observation_space = rltorch.Box(self.low, self.high)
end

function MountainCar_CompleteSensor:observe(world)
  return world.state:reshape(1,2)
end