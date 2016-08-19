  local Atari_ImageSensor = torch.class('rltorch.Atari_ImageSensor','rltorch.Sensor'); 
    
--- returns a 2D vector with position and speed
function Atari_ImageSensor:__init(world) 
  
 self.observation_space = rltorch.Box(0,255,torch.LongStorage({3,world.HEIGHT,world.WIDTH}))
 self.obs = torch.ByteTensor(world.HEIGHT,world.WIDTH)    
 self.fobs=torch.Tensor(3,world.HEIGHT,world.WIDTH)    
end

function Atari_ImageSensor:observe(world)
  world.ale:fillObs(torch.data(self.obs), self.obs:nElement())    
  local observation=alewrap.getRgbFromPalette(self.obs) 
  self.fobs:copy(observation):div(255)
  return self.fobs
end
