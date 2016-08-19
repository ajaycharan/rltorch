 
local Atari_Task = torch.class('rltorch.Atari_Task','rltorch.Feedback'); 
   
  
function Atari_Task:__init(world)
  self.world=world
  print("la")
end

function Atari_Task:feedback(world)
    return {reward=world.reward}
end

function Atari_Task:finished(world)  
  local done=world.ale:isGameOver()
  return done
end
