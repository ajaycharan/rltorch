
local MountainCarWorld = torch.class('rltorch.MountainCarWorld','rltorch.World'); 
  

function MountainCarWorld:__init(parameters)
  rltorch.World.__init(self,parameters)
  
  self.min_position = -1.2
  self.max_position = 0.6
  self.max_speed = 0.07
  
  self.low = torch.Tensor({self.min_position, -self.max_speed})
  self.high = torch.Tensor({self.max_position, self.max_speed})
  self.viewer = None

  self.action_space = rltorch.Discrete(3)
end
 

function MountainCarWorld:step(agent_action)  
  local position=self.state[1]
  local velocity=self.state[2]
  velocity = velocity+ (agent_action-2)*0.001 + math.cos(3*position)*(-0.0025)
  if (velocity > self.max_speed) then velocity = self.max_speed end
  if (velocity < -self.max_speed)then velocity = -self.max_speed end
  position = position + velocity
  if (position > self.max_position)then position = self.max_position end
  if (position < self.min_position)then position = self.min_position end
  if (position==self.min_position and velocity<0) then velocity = 0 end  
  self.state[1] = position
  self.state[2] = velocity
end


function MountainCarWorld:reset()
   self.state = torch.Tensor({math.random()*(-0.4+0.6)-0.4, math.random()*(2*self.max_speed)-self.max_speed})   
end 


function MountainCarWorld:close()
end
