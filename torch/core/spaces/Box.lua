  
 --- Describe a sequential environment with one or many agents
local Box = torch.class('rltorch.Box','rltorch.Space'); 
 
--- Initialize the environment
--- if shape is a tensor (or longtensor)
  --- min is the min value and max if the max value (rela) over all the dimensions
--- else low and high must have the same shape
--- shape is optionnal
function Box:__init(low,high,shape)
  if (shape~=nil) then
    self.low=torch.Tensor(shape):fill(low)
    self.high=torch.Tensor(shape):fill(high)
  else    
    self.low=low
    self.high=high  
  end
end
 
---Update the environment given that one agent has chosen one action
-- @params agent_action the action of the agent
-- @returns observation,reward,done,info
function Box:sample()
  local r=torch.rand(self.low:size())
  r:cmul(self.high-self.low)
  return r+self.low
end

---- Returns the initial domain 
-- @return the action domain
function Box:contains(x)
  local overmax=torch.gt(x,self.high):sum()
  if (overmax>0) then return(false) end
  
  local undermin=torch.lt(x,self.low):sum()
  if (undermin>0) then return(false) end
  
  return(true)
end 

function Box:size()
  return(self.low:size())
end

function Box:toJSON()
  local js={name="Box",low=torch.totable(self.low),high=torch.totable(self.high)}
  return(json.encode(js))
end

function Box:convertValueToString(x,format)
  if (format=="json") then
    local t=torch.totable(x)
    return(json.encode(t))
  elseif (format=="torch") then
    return(torch.serialize(x)) 
  else
    assert(false)
    return nil
  end  
end

function Box:convertStringToValue(_str,format)
  if (format=="json") then
    local t=json.decode(_str)
    return(torch.Tensor(t))
  elseif(format=="torch") then
    return torch.deserialize(_str)
  else
    assert(false)
    return nil
  end  
end


 
