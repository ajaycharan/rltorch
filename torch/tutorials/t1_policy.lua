 
require('rltorch')

-- Create the world
world = rltorch.CartPoleWorld()

-- Create the learning problem
task=rltorch.CartPole_Task0(world)

-- Create the sensor used by the agent
sensor=rltorch.CartPole_CompleteSensor(world)

-- Create the sensor that will be used for visualization
renderer=rltorch.CartPole_QtSensor(world,320,200)

-- Create a random policy
policy=rltorch.RandomPolicy(nil,world.action_space)

MAX_LENGTH=100
DISCOUNT_FACTOR=0.9

for i=1,100 do
    print("Starting episode "..i)
    
    world:reset()
    
    -- Compute an observation of the world and feed the policy
    local observation=sensor:observe(world)
    policy:new_episode(observation)  
    
    local sum_reward=0.0
    local current_discount=1.0
    
    for t=1,MAX_LENGTH do        
      local action=policy:sample()      
      world:step(action)
      local done=task:finished(world)
      
      local feedback=task:feedback(world)
      assert(feedback.reward~=nil)
      policy:feedback(feedback) -- the immediate reward is provided to the policy
      
      -- Feed the policy with a new observation
      observation=sensor:observe(world)
      policy:observe(observation)      
      
      -- Computes the overall reward
      sum_reward=sum_reward+current_discount*feedback.reward -- computes the discounted sum of rewards
      current_discount=current_discount*DISCOUNT_FACTOR      
      if (done) then
        break
      end 
    end
    policy:end_episode(sum_reward) -- The feedback provided for the whole episode here is the discounted sum of rewards   
    print("Reward is "..sum_reward)
end
world:close()