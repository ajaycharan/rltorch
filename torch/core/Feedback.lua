 
 -- The Feedback class provides feedback to the agent(s) acting in a world. Feedback thus designs a Learning Problem over a particular world
local Feedback = torch.class('rltorch.Feedback');  

function Feedback:__init(world)
end

function Feedback:feedback(world)
  assert(false)
end

function Feedback:finished(world)
  return(false)
end