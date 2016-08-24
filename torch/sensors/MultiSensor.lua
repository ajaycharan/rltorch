 local MultiSensor = torch.class('rltorch.MultiSensor','rltorch.Sensor'); 

 -- A sensor that returns a table of observation given a table of sensors
function MultiSensor:__init(world,sensors)
  self.sensors=sensors  
  self.observation_space=rltorch.MultipleSpaces()
  for a=1,#self.sensors do 
    self.observation_space:add(self.sensors[a].observation_space)
  end
end

function MultiSensor:observe(world)  
 local retour={}
 for a=1,#self.sensors do retour[a]=self.sensors[a]:observe(world) end
 return retour
end

 function MultiSensor:qtDisplay(observations)
   if (self.windows==nil) then self.windows={} end
   for a=1,#self.sensors do
       self.windows[a] = image.display({image=observations[a], win=self.windows[a]})
    end
end
