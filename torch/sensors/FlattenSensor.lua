 local FlattenSensor = torch.class('rltorch.FlattenSensor','rltorch.Sensor'); 

 
function FlattenSensor:__init(world,sensor_to_flatten)
  self.sensor_to_flatten=sensor_to_flatten
  local s=sensor_to_flatten.observation_space
  assert(torch.type(s)=="rltorch.Box")
  
  self.n=s.low:nElement()
  self.observation_space=rltorch.Box(s.low:reshape(1,self.n),s.high:reshape(1,self.n))
end

function FlattenSensor:observe(world)  
 local o=self.sensor_to_flatten:observe(world):reshape(1,self.n)
 print(o:size())
   return o
end

 
