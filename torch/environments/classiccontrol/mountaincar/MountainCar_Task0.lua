
local MountainCar_Task0 = torch.class('rltorch.MountainCar_Task0','rltorch.Feedback'); 
   
  
function MountainCar_Task0:__init(world)
  self.goal_position = 0.5
end

function MountainCar_Task0:feedback(world)
  return {reward=-1}
end

function MountainCar_Task0:finished(world)  
  local done = (world.state[1]>= self.goal_position)
  return done
end

