 local OpenAISensor = torch.class('rltorch.OpenAISensor','rltorch.Sensor'); 

 
function OpenAISensor:__init(observation_space)
  
  local s=observation_space
  assert(torch.type(s)=="rltorch.Box")
  
  self.n=s.low:nElement()
  self.observation_space=rltorch.Box(s.low:reshape(1,self.n),s.high:reshape(1,self.n))
end

function OpenAISensor:size()
  return self.n
end

function OpenAISensor:observe(observation)  
 local o=observation:reshape(1,self.n)
 return o
end

 
