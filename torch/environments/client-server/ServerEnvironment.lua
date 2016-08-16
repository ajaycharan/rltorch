require 'json'
--- Transform a classical environment to a server-based environment

local ServerEnvironment = torch.class('rltorch.ServerEnvironment'); 
   
function ServerEnvironment:__init(environment,port,format,_render,_render_arg)        
  self.environment=environment      
  self.port=port
  self._render=_render
  self._render_arg=_render_arg
  self.format=format
  self:waitConnection()
  self:main()
end

function ServerEnvironment:waitConnection()        
  
  print("== Launching server on port "..self.port)
  self.socket=require('socket')
  self.server=assert(socket.bind("*",self.port))
  print("== Waiting for a connection...")
  self.client=self.server:accept()
  local line,err=self.client:receive()
  local name=json.decode(line)
  assert(name.name~=nil)
  print("Client name is :\n"..name.name)
  print("== Sending action space and observation space descriptions")
  local as=self.environment.action_space
  local os=self.environment.observation_space
  self.client:send(as:toJSON().."\n")
  self.client:send(os:toJSON().."\n")
  print("=== Ready to run")
end

function ServerEnvironment:main()
  local flag=true
  while(flag) do
        
    local command,err=self.client:receive()        
    command=json.decode(command)
    if (command.command=="r") then
      obs=self.environment:reset(command.parameters)   
      
      local str_obs=self.environment.observation_space:convertValueToString(obs,self.format)
      self.client:send(str_obs.."\n")      
    elseif (command.command=="s") then
      local agent_action=self.environment.action_space:convertStringToValue(command.agent_action,self.format)
      local observation,reward,done,info=unpack(self.environment:step(agent_action))
      
      if (self.format=="json") then
        local str_observation=self.environment.observation_space:convertValueToString(observation,self.format)
        local _json=json.encode({str_observation,reward,done,info})      
        self.client:send(_json.."\n")        
      else
        local str_observation=self.environment.observation_space:convertValueToString(observation,self.format)
        self.client:send(tostring(string.len(str_observation)).."\n")
        self.client:send(str_observation)
        local _json=json.encode({reward,done,info})      
        self.client:send(_json.."\n")
      end
    elseif (command.command=="c") then 
      self:close()
      flag=false
    end
    if (self._render) then self:render(self._render_arg) end
  end
end
 
function ServerEnvironment:step(agent_action)  
  return self.environment:step(agent_action)
   
end


function ServerEnvironment:reset(parameters)
    local a=self.environment:reset(parameters)
    return a
end 

function ServerEnvironment:close()
  self.environment:close()
end

function ServerEnvironment:render(arg)
  self.environment:render(arg)
end

