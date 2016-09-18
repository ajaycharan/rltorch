
require 'gnuplot'

local ExperimentLog = torch.class('rltorch.ExperimentLog'); 

function ExperimentLog:__init(memory)
  self.memory=memory
  self.jsons={}
  self.currentjson=nil
  self.iteration=0
  self.parameters={}
end

function ExperimentLog:addFixedParameters(arg)
    for k,v in pairs(arg) do
      self.parameters[k]=v
    end
end

function ExperimentLog:isEmpty()
  if (self.currentjson==nil) then return true end
  local nb=0
  for k,v in pairs(self.currentjson) do
    nb=nb+1
  end
  if (nb==0) then return true end
  return false
end

function ExperimentLog:newIteration()
  if (self.memory) then
    if (self.currentjson~=nil) then
      table.insert(self.jsons,self.currentjson)
    end
  end
  self.iteration=self.iteration+1
  self.currentjson={}
end

function ExperimentLog:addValue(key,value)
  self.currentjson[key]=value
end

function ExperimentLog:addDescription(text)  
end

function ExperimentLog:size()
  return table.getn(self.jsons)
end

function ExperimentLog:getColumn(name)
  local s=#self.jsons
  local c=torch.Tensor(s)
  for k,v in ipairs(self.jsons) do
    c[k]=v[name]
  end
  return c
end

------ each element is {name=name of the column, type=single, cumsum or sliding, size=size of sliding window, mul can be a multiplicative coefficient}
function ExperimentLog:plot(to_plot)  
  if (#self.jsons==0) then return end
  local ns=to_plot; print(ns)
  local tt={}
  local pos=1  
  for k,v in pairs(ns) do
    if ((v.type==nil) or (v.type=="single")) then
      tt[pos]={v.name,self:getColumn(v.name),"linespoints ls "..k}
      
    elseif(v.type=="cumsum") then
      tt[pos]={v.name.." (cumsum)",self:getColumn(v.name):cumsum(),"linespoints ls "..k}
      
    elseif(v.type=="avg_cumsum") then
      local c=self:getColumn(v.name):cumsum()
      local d=torch.linspace(1,c:size(1),c:size(1))
      c:cdiv(d)
      tt[pos]={v.name.." (avg_cumsum)",c,"linespoints ls "..k}
      
    elseif(v.type=="avg_sliding") then
      local c=self:getColumn(v.name)
      local s=v.size
      local t
      if (c:size(1)>s) then
        local cs=c:cumsum()
        t=torch.Tensor(c:size(1))
        for i=1,s do
          t[i]=cs[i]/i
        end
        for i=s+1,c:size(1) do
          t[i]=c:narrow(1,i-s,s):sum()/s
        end
      else
        t=c:cumsum()
        t:cdiv(torch.linspace(1,c:size(1),c:size(1)))
      end
      tt[pos]={v.name.." (avg_sliding "..s..")",t}
    end
    ------
    if (v.mul~=nil) then tt[pos][2]:mul(v.mul) end
    
    
    
    pos=pos+1
  end
  print(tt)
  gnuplot.plot(tt)
end

function ExperimentLog:plot3D(columns)  
  if (#self.jsons==0) then return end
  local tt={}; local pos=1
  for _,c in ipairs(columns) do    
    for k,v in ipairs(self.jsons) do
      local value=v[c.name]
      assert(type(value)=='table')
      local vector=torch.Tensor(value)
      if (c.type=="cumsum") then vector=vector:cumsum() end
      tt[pos]={c.name.." at "..k,vector,"lines ls "..pos}
      pos=pos+1
    end
  end
  gnuplot.plot(tt)
end

function ExperimentLog:close()
end

