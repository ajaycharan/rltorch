local MulticlassClassification_Sensor = torch.class('rltorch.MulticlassClassification_Sensor','rltorch.Sensor'); 

            
    
--- returns a 2D vector with position and speed
function MulticlassClassification_Sensor:__init(world)  
   local vmin=world.parameters.training_examples:min(1)[1]:reshape(1,world.n)
   local vmax=world.parameters.training_examples:max(1)[1]:reshape(1,world.n)
   self.observation_space = rltorch.Box(vmin,vmax)
end

function MulticlassClassification_Sensor:observe(world)  
  if (world.test) then
    if (world.current_index>world.parameters.testing_examples:size(1)) then return nil end
    return world.parameters.testing_examples[world.current_index]:reshape(1,world.n)
  else
    return world.parameters.training_examples[world.current_index]:reshape(1,world.n)
  end
end 
