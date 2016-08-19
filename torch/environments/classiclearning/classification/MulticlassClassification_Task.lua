 local MulticlassClassification_Task = torch.class('rltorch.MulticlassClassification_Task','rltorch.Feedback'); 
   
   
function MulticlassClassification_Task:__init(world)
  
end

function MulticlassClassification_Task:feedback(world)
  if (world.test) then
    local true_category=world.parameters.testing_labels[world.last_index][1]          
    local feed={true_action=true_category,reward=0}          
    if (true_category==world.last_action) then feed.reward=1 end
    return feed
  else
    local true_category=world.parameters.training_labels[world.last_index][1]
    local feed={true_action=true_category,reward=0}
    if (true_category==world.last_action) then feed.reward=1 end
    return feed
  end
end

function MulticlassClassification_Task:finished(world)  
  if (world.test) then
    if (world.current_index>world.parameters.testing_examples:size(1)) then return true else return false end    
  else
    return false
  end
end


   

