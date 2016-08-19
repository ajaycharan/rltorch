 
require('rltorch')


-- Create the world
world=rltorch.CartPoleWorld()

-- Create a sensor for rendering
renderer=rltorch.CartPole_QtSensor(world,320,200)

-- One trajectory
world:reset()
for i=1,1000 do    
    -- Apply a random action
    world:step(world.action_space:sample())
    
    -- Use the sensor to visualize the world state
    renderer:observe(world)
    sys.sleep(0.1)
end