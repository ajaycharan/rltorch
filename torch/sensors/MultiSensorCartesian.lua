 local MultiSensorCartesian = torch.class('rltorch.MultiSensorCartesian','rltorch.Sensor'); 

 -- A sensor that returns a concatenation of the observation of multiple sensors (as a 1xn vector)
function MultiSensorCartesian:__init(world,sensors)
  self.sensors=sensors  
  self.n={}
  self.ntotal=0
  local cat_min,cat_max  
  for a=1,#sensors do
    local s=sensors[a].observation_space
    assert(torch.type(s)=="rltorch.Box")
    self.n[a]=s.low:nElement()
    self.ntotal=self.ntotal+self.n[a]
    
    if (a==1) then 
      cat_min=s.low:reshape(self.n[a])
      cat_max=s.high:reshape(self.n[a])
    else
      cat_min=torch.cat(cat_min,s.low:reshape(self.n[a]))
      cat_max=torch.cat(cat_max,s.high:reshape(self.n[a]))
    end
  end  
  self.observation_space=rltorch.Box(cat_min:reshape(1,cat_min:size()[1]),cat_max:reshape(1,cat_min:size()[1]))
end

function MultiSensorCartesian:observe(world)  
 local retour 
 for a=1,#self.sensors do 
   if (a==1) then retour=self.sensors[a]:observe(world):reshape(self.n[a]) 
 else retour=torch.cat(retour,self.sensors[a]:observe(world):reshape(self.n[a])) end
 end
 return retour:reshape(1,retour:size(1))
end

 
 
