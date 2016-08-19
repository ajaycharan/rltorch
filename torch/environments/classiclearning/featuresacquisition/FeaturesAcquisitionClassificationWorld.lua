local FeaturesAcquisitionClassificationWorld = torch.class('rltorch.FeaturesAcquisitionClassificationWorld','rltorch.World'); 
  
 
function FeaturesAcquisitionClassificationWorld:__init(parameters) 
        rltorch.World.__init(self,parameters)
      
        assert(self.parameters.training_examples~=nil)
        assert(self.parameters.training_labels~=nil)
        self.n=self.parameters.training_examples:size(2)
        self.action_space = rltorch.Discrete(self.n)
        self.nb_train_examples=self.parameters.training_examples:size(1)
        self.test=false
        self.current_data_idx=0
end
 
function FeaturesAcquisitionClassificationWorld:step(agent_action)  
        self.last_action=agent_action        
end

function FeaturesAcquisitionClassificationWorld:reset(use_test)    
    if (use_test) then
      if (self.test==true) then self.current_data_idx=self.current_data_idx+1 else self.current_data_idx=1 end
      if (self.current_data_idx>self.parameters.testing_examples:size(1)) then self.current_data_idx=1 end
      self.test=true
    else
      self.test=false
      self.current_data_idx=math.random(self.parameters.training_examples:size(1))
    end
    self.last_action=0
end

function FeaturesAcquisitionClassificationWorld:close()
end

 
 
