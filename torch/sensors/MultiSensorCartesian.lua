 local MultiSensorCartesian = torch.class('rltorch.MultiSensorCartesian','rltorch.Sensor'); 

 -- A sensor that returns a concatenation of the observation of multiple sensors (1*n_i) => (as a 1x(n_1*n_2*...*n_m) vector)
function MultiSensorCartesian:__init(world,sensors)
  self.sensors=sensors  
  self.n={}
  self.ntotal=1
  local cat_min=torch.Tensor(1,1):fill(1)
  for a=1,#sensors do
    local s=sensors[a].observation_space
    assert(torch.type(s)=="rltorch.Box")
    self.n[a]=s.low:nElement()
    self.ntotal=self.ntotal*self.n[a]
  end    
    
  self.observation_space=rltorch.Box(torch.Tensor(1,self.ntotal):fill(0),torch.Tensor(1,self.ntotal):fill(1))
end

function MultiSensorCartesian:observe(world)  
 local retour=torch.Tensor(1,1):fill(1)
 for a=1,#self.sensors do 
   retour=retour:transpose(1,2)*self.sensors[a]:observe(world):reshape(1,self.n[a]) 
   retour=retour:reshape(1,retour:nElement())
 end
 return retour
end

 
 
