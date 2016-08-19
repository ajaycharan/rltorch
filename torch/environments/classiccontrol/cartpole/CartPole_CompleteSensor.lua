local CartPole_CompleteSensor = torch.class('rltorch.CartPole_CompleteSensor','rltorch.Sensor'); 
    
--- returns a 2D vector with position and speed
function CartPole_CompleteSensor:__init(world)  
 local high = torch.Tensor({{100000, 100000, 100000, 100000}})
 self.action_space = rltorch.Discrete(2)
 self.observation_space = rltorch.Box(-high, high) 
end

function CartPole_CompleteSensor:observe(world)
  return world.state:reshape(1,4)
end
        
        -- Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        