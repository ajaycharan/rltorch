local MulticlassClassificationWorld = torch.class('rltorch.MulticlassClassificationWorld','rltorch.World'); 
  
 
function MulticlassClassificationWorld:__init(parameters) 
        rltorch.World.__init(self,parameters)
      
        assert(self.parameters.training_examples~=nil)
        assert(self.parameters.training_labels~=nil)
        local vmax,imax=self.parameters.training_labels:max(1)
        self.action_space = rltorch.Discrete(vmax[1][1])
        self.n=self.parameters.training_examples:size(2)
        self.nb_train_examples=self.parameters.training_examples:size(1)
        self.test=false
end
 
function MulticlassClassificationWorld:step(agent_action)  
        self.last_action=agent_action
          --- TESTING MODE (in test, the next vector is the next in the  testing set)
        if (self.test) then                    
          self.last_index=self.current_index
          self.current_index=self.current_index+1          
        else ---- TRAINING MODEL (in train, the next vector is a random one)
          self.last_index=self.current_index
          self.current_index=math.random(self.nb_train_examples)          
        end
end


function MulticlassClassificationWorld:reset(use_test)    
    if (use_test==true) then
      self.test=true
      self.last_index=nil
      self.current_index=1      
    else
      self.test=false
      self.last_index=nil
      self.current_index=math.random(self.nb_train_examples)            
    end
end 

function MulticlassClassificationWorld:close()
end

 
