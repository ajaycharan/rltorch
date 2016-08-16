require 'json'
 --- Describe an (observation/action) space
local Space = torch.class('rltorch.Space'); 
 
function Space:__init()
end
 
---- Sample one element of the space with a uniform distribution
function Space:sample()
  assert(false,"Space:sample")
end

---- Returns true of the space contains this element
function Space:contains(x)
   assert(false,"Space:contains")
end 
 
function Space:toJSON()
  assert(false)
end

function Space:buildFromJSON(_json)
  local _j=json.decode(_json)
  if (_j.name=="Box") then
    return(rltorch.Box(torch.Tensor(_j.low),torch.Tensor(_j.high)))
  elseif (_j.name=="Discrete") then
    return(rltorch.Discrete(_j.n))  
  else 
    assert(false,"Unable to build a space from "..json.encode(_json))
  end    
end

function Space:convertValueToString(x,format)
  assert(false)
end

function Space:convertStringToValue(_str,format)
  assert(false)
end