local FeaturesAcquisitionClassification_Sensor = torch.class('rltorch.FeaturesAcquisitionClassification_Sensor','rltorch.Sensor'); 

            
    
--- returns a 2D vector with position and speed
function FeaturesAcquisitionClassification_Sensor:__init(world)  
   local vmin=torch.Tensor({{world.parameters.training_examples:min()}})
   local vmax=torch.Tensor({{world.parameters.training_examples:max()}})
   self.observation_space = rltorch.Box(vmin,vmax)    
end

function FeaturesAcquisitionClassification_Sensor:observe(world)  
  if (world.last_action<1) then return torch.Tensor(1,1):fill(0) end
  
  if (world.test) then
    return world.parameters.testing_examples[world.current_data_idx][world.last_action]:reshape(1,1)   
  else
    return world.parameters.training_examples[world.current_data_idx][world.last_action]:reshape(1,1)
  end
end 
