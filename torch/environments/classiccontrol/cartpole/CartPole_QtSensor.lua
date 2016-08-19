local CartPole_QtSensor = torch.class('rltorch.CartPole_QtSensor','rltorch.Sensor'); 
    
--- Draw the state of MC into a window
function CartPole_QtSensor:__init(world,sizex,sizey)  
  self.observation_space = nil
  self.sizex=sizex
  self.sizey=sizey  
end

function CartPole_QtSensor:observe(world)
    local SX=self.sizex
    local SY=self.sizey
    local POLE_SIZE=SY/5
    
    if (self.__render_widget==nil) then 
      require 'qt'
      require 'qtuiloader'
      require 'qtwidget'
      
      self.__render_widget = qtwidget.newwindow(SX,SY,"CatPole_v0")
    end
    local CART_SX=SX/20
    local CART_SY=SY/20
    self.__render_widget:setcolor("white")
      
    self.__render_widget:showpage()
    self.__render_widget:stroke()
    self.__render_widget:setcolor("red")
    local x=world.state[1]*10
    local VX=x+SX/2
    local VY=SY/2
    self.__render_widget:rectangle(VX-CART_SX/2,VY-CART_SY/2,CART_SX,CART_SY)
    self.__render_widget:fill()
    
    local angle=world.state[3]
    local pole_x=math.sin(angle)*POLE_SIZE
    local pole_y=-math.cos(angle)*POLE_SIZE
    
    self.__render_widget:setlinewidth(5)
    self.__render_widget:setcolor("blue")
    self.__render_widget:moveto(VX,VY)
    self.__render_widget:lineto(VX+pole_x,VY+pole_y)    
    self.__render_widget:stroke()
    self.__render_widget:painter()
end

 
