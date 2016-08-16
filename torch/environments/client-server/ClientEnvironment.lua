require 'json'
--- Create an environment from a socet

local ClientEnvironment = torch.class('rltorch.ClientEnvironment','rltorch.Environment'); 
   
  
function ClientEnvironment:__init(host,port,format,name)        
  self.port=port
  self.host=host
  self.name=name
  self.format=format
  self:doConnection()  
end

function ClientEnvironment:doConnection()        
  
  print("== Connecting to "..self.host.." on port "..self.port)
  self.socket=require('socket')
  local ip = assert(socket.dns.toip(self.host))
  print("Ip is "..ip)
  self.client=assert(socket.connect(ip,self.port))
  if(self.name==nil) then self.name={name="NoName"} else self.name={name=self.name} end  
  self.client:send(json.encode(self.name).."\n")
  
  -- now we receive action space and observation space
  print("== Receiving action and observation space description")
  local line,err=self.client:receive()
  self.action_space=rltorch.Space():buildFromJSON(line)
  line,err=self.client:receive()
  self.observation_space=rltorch.Space():buildFromJSON(line)
  print(self.action_space)
  print(self.action_space.n)
  print(self.observation_space)
  print("=== Ready to run")
end
 
function ClientEnvironment:step(agent_action) 
  local tos={command="s",agent_action=self.action_space:convertValueToString(agent_action,self.format)}
  self.client:send(json.encode(tos).."\n")
  if (self.format=="json") then
    local sfe,err=self.client:receive()    
    local ffl=json.decode(sfe)
    return({self.observation_space:convertStringToValue(ffl[1],self.format),ffl[2],ffl[3],ffl[4]})
  else
    local sfe,err=self.client:receive()
    local ss=tonumber(sfe)
    local fe,err=self.client:receive(ss)
    local fl,err=self.client:receive()
    local ffl=json.decode(fl)
    return({self.observation_space:convertStringToValue(fe,self.format),ffl[1],ffl[2],ffl[3]})
  end
--  return({nil,ffe[2],ffe[3],ffe[4]})
end

function ClientEnvironment:reset(parameters)
  local tos={command="r",parameters=parameters}
  self.client:send(json.encode(tos).."\n")
  local obs,err=self.client:receive()
  return(self.observation_space:convertStringToValue(obs,self.format))
end 

function ClientEnvironment:close()
  self.client:send("{\"command\":\"c\"}\n")
end

function ClientEnvironment:render(arg)
  
end

