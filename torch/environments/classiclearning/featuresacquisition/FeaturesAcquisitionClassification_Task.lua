 local FeaturesAcquisitionClassification_Task = torch.class('rltorch.FeaturesAcquisitionClassification_Task','rltorch.Feedback'); 
   
   
function FeaturesAcquisitionClassification_Task:__init(world)
  
end

function FeaturesAcquisitionClassification_Task:feedback(world)
  if (world.test) then
   local true_category=world.parameters.testing_labels[world.current_data_idx][1]          
    local feed={target=torch.Tensor({true_category})}    
    return feed
  else
    local true_category=world.parameters.training_labels[world.current_data_idx][1]          
    local feed={target=torch.Tensor({true_category})}    
    return feed
  end
end

function FeaturesAcquisitionClassification_Task:finished(world)  
  return false  
end


   

