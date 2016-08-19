
local CartPole_Task0 = torch.class('rltorch.CartPole_Task0','rltorch.Feedback'); 
   
  
function CartPole_Task0:__init(world)
  self.theta_threshold_radians = 12 * 2 * math.pi / 360
  self.x_threshold = 2.4
end

function CartPole_Task0:feedback(world)
    return {reward=1}
end

function CartPole_Task0:finished(world)  
  local done =  world.state[1] < -self.x_threshold 
                or world.state[1] > self.x_threshold 
                or world.state[3] < -self.theta_threshold_radians 
                or world.state[3] > self.theta_threshold_radians
  return done
end
