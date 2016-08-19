
local CartPole_v0 = torch.class('rltorch.CartPole_v0','rltorch.Environment'); 
  
function CartPole_v0:__init()
  local world=rltorch.CartPoleWord()
  rltorch.Environment.__init(self,world,rltorch.CartPole_CompleteSensor(world),rltorch.CartPole_Task0(world))
end
 
