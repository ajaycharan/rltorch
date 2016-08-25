
-- This class describes a trajectory (o1, a1, r1, o2, a2, r2, o3, ....)
local Trajectory = torch.class('rltorch.Trajectory'); 

function Trajectory:__init()
  self.observations={}
  self.actions={}
  self.feedback={}
  self.done={}
end

function Trajectory:push_observation(o)  
  self.observations[#self.observations+1]=o
end
  
function Trajectory:push_action(o)
  self.actions[#self.actions+1]=o
end
  
function Trajectory:push_feedback(o)
  self.feedback[#self.feedback+1]=o
end
  
function Trajectory:push_done(o)
  self.done[#self.done+1]=done
end

function Trajectory:get_number_of_observations()
  return( #self.observations)
end
  
