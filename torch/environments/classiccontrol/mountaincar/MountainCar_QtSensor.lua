local MountainCar_QtSensor = torch.class('rltorch.MountainCar_QtSensor','rltorch.Sensor'); 
    
--- Draw the state of MC into a window
function MountainCar_QtSensor:__init(world,sizex,sizey)  
  self.observation_space = nil
  self.sizex=sizex
  self.sizey=sizey
  
end

function MountainCar_QtSensor:_height(xs)
  return math.sin(3 * xs)*0.45
end

function MountainCar_QtSensor:observe(world)
    local SX=self.sizex
    local SY=self.sizey
    local CAR_SIZE=SY/5
    if (self.__render_widget==nil) then 
        require 'qt'
        require 'qtuiloader'
        require 'qtwidget'
        
        self.__render_widget = qtwidget.newwindow(SX,SY,"MountainCar")
        self.__render_widget:setangleunit("Degrees")
    end
      
    ---
    local scale_y=SY*0.5
    self.__render_widget:showpage()
    self.__render_widget:setcolor("black")
    do
      local pos=world.min_position
      local py=SY/2*scale_y+self:_height(pos)
      self.__render_widget:fill(false)  
      self.__render_widget:moveto(1,py)
      self.__render_widget:stroke()
      for px=2,SX,10 do
          pos=px/SX*(world.max_position-world.min_position)+world.min_position
          py=SY/2-scale_y*self:_height(pos)
          self.__render_widget:lineto(px,py)
      end
      self.__render_widget:stroke()      
      
      self.__render_widget:setcolor("red")
      local ppx=world.state[1]
      local pos=((ppx-world.min_position)/(world.max_position-world.min_position))*SX
      local ppy=SY/2-scale_y*self:_height(ppx)
      
      self.__render_widget:arc(math.floor(pos)+1,math.floor(ppy),SX/100.0,0,360)
      self.__render_widget:fill(true)  
      self.__render_widget:stroke()      
    end    
    self.__render_widget:painter()      
end

