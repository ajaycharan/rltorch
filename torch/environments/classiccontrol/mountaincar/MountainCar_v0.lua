
local MountainCar_v0 = torch.class('rltorch.MountainCar_v0','rltorch.Environment'); 
  

function MountainCar_v0:__init()
  local world=rltorch.MountainCarWorld()
  rltorch.Environment.__init(self,world,rltorch.MountainCar_CompleteSensor(world),rltorch.MountainCar_Task0(world))
end
 
