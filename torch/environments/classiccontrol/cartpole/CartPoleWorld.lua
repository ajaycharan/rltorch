
local CartPoleWorld = torch.class('rltorch.CartPoleWorld','rltorch.World'); 
  
 
function CartPoleWorld:__init(parameters)
        rltorch.World.__init(self,parameters)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 -- actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  -- seconds between state updates
        self.action_space=rltorch.Discrete(2)
end
 
function CartPoleWorld:step(agent_action)  
        local action = agent_action
        assert(action==1 or action==2, "Invalid action")
        local x=self.state[1]
        local x_dot=self.state[2]
        local theta=self.state[3] 
        local theta_dot = self.state[4]
        
        local force=0
        if (action==2) then force = self.force_mag else force=-self.force_mag end
        local costheta = math.cos(theta)
        local sintheta = math.sin(theta)
        local temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        local thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        local xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state[1]=x
        self.state[2]=x_dot
        self.state[3]=theta
        self.state[4]=theta_dot
end


function CartPoleWorld:reset()
    self.state = torch.rand(4)*0.1-torch.Tensor(4):fill(-0.05)
end 

function CartPoleWorld:close()
end

