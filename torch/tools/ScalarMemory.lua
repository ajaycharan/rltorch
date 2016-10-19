local ScalarMemory = torch.class('rltorch.ScalarMemory'); 

function ScalarMemory:__init(size_memory)
  self.size_memory=size_memory
  self.INIT_SIZE=10000
  
  if (self.size_memory==nil) then      
    self.memory=torch.Tensor(self.INIT_SIZE)
  else
    self.memory=torch.Tensor(self.size_memory)
  end
  self.position_in_memory=0
  self.memory_is_full=false
end

function ScalarMemory:push(value)
  if (self.size_memory==0) then return end
  self.position_in_memory=self.position_in_memory+1
  
  if (self.size_memory==nil) then 
    if (self.position_in_memory>self.memory:size(1)) then
      self.memory:resize(self.memory:size(1)+self.INIT_SIZE)
    end
    self.memory[self.position_in_memory]=value
    return 
  end
  
  if (self.position_in_memory>self.size_memory) then
    self.position_in_memory=1
    self.memory_is_full=true
  end
  self.memory[self.position_in_memory]=value
end

function ScalarMemory:mean()
  if (self.size_memory==0) then return end
  if (self.memory_is_full) then return self.memory:mean() end
  
  return self.memory:narrow(1,1,self.position_in_memory):mean()
end

function ScalarMemory:get_last() 
    if (self.size_memory==0) then return end

  return self.memory[self.position_in_memory]
end

function ScalarMemory:getMemory()
  if (self.size_memory==0) then return torch.Tensor() end

  if (not self.memory_is_full) then
    return self.memory:narrow(1,1,self.position_in_memory)
  end
  
  local r=torch.Tensor(self.size_memory)
  local pos=self.position_in_memory+1
  if (pos>self.size_memory) then r:copy(self.memory) return r end
  r:narrow(1,1,self.size_memory-pos+1):copy(self.memory:narrow(1,pos,self.size_memory-pos+1))
  r:narrow(1,self.size_memory-pos+2,self.position_in_memory):copy(self.memory:narrow(1,1,self.position_in_memory))
  return r
end

function ScalarMemory:getSlidingMean(ws)
  if (self.size_memory==0) then return torch.Tensor() end

  local r=self:getMemory()
  if (r:size(1)<ws) then 
    local rr=torch.Tensor(r:size(1)) for k=1,r:size(1) do rr[k]=r:narrow(1,1,k):mean() end    
    return rr 
  end
  
  local rr=torch.Tensor(r:size(1))
  for k=1,ws do rr[k]=r:narrow(1,1,k):mean() end 
  for i=1,r:size(1)-ws do
    rr[ws+i]=r:narrow(1,i,ws):mean()
  end
  return rr
end

