require 'torch'

local AtariWorld_Breakout = torch.class('rltorch.AtariWorld_Breakout','rltorch.World'); 


function AtariWorld_Breakout:__init(parameters)
  rltorch.World.__init(self,parameters)
  assert(self.parameters.rom~=nil)
  self.HEIGHT=210
  self.WIDTH=160
  self.RAM_LENGTH=128
  
  self.config = {
        gameOverReward=0,
        enableRamObs=false,
    }
  self.win = nil
  self.ale = alewrap.newAle(self.parameters.rom)  
  self.action_space = rltorch.Discrete(3)
end

 
function AtariWorld_Breakout:reset()
  local v=self.ale:resetGame()  
end  


function AtariWorld_Breakout:step(agent_action)
  if (agent_action==1) then agent_action=4 end
  if (agent_action==2) then agent_action=5 end
  if (agent_action==3) then agent_action=2 end
  self.reward = self.ale:act(agent_action-1)  
end



function AtariWorld_Breakout:close()
  
end

--function Atari_v0:render(arg)
--  if (arg.mode=="console") then
--    print("Cannot render Atari in console")
--  elseif (arg.mode=="qt") then
--    self.win=image.display({image=self.last_observation,win=self.win})
--  end
--end

--ACTION_MEANING = {
--    0 : "NOOP",
--    1 : "FIRE",
--    2 : "UP",
--    3 : "RIGHT",
--    4 : "LEFT",
--    5 : "DOWN",
--    6 : "UPRIGHT",
--    7 : "UPLEFT",
--    8 : "DOWNRIGHT",
--    9 : "DOWNLEFT",
--    10 : "UPFIRE",
--    11 : "RIGHTFIRE",
--    12 : "LEFTFIRE",
--    13 : "DOWNFIRE",
--    14 : "UPRIGHTFIRE",
--    15 : "UPLEFTFIRE",
--    16 : "DOWNRIGHTFIRE",
--    17 : "DOWNLEFTFIRE",
--}
