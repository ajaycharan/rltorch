require 'torch'
rltorch={}

include('World.lua')
include('Environment.lua')
include('Feedback.lua')
include('Sensor.lua')

include('RLTools.lua')
include('RLFile.lua')

include('Trajectory.lua')
include('Trajectories.lua')

include('FlattenSensor.lua')

include('Space.lua')
include('Discrete.lua')
include('Box.lua')
include('MultipleSpaces.lua')

-- Mountain car
	include('MountainCarWorld.lua')
	include('MountainCar_CompleteSensor.lua')
	include('MountainCar_QtSensor.lua')
	include('MountainCar_Task0.lua')
	include('MountainCar_v0.lua')

-- Cart Pole
	include('CartPoleWorld.lua')
	include('CartPole_CompleteSensor.lua')
	include('CartPole_QtSensor.lua')
	include('CartPole_Task0.lua')
	include('CartPole_v0.lua')

-- Atari 
	include('AtariWorld.lua')
	include('Atari_Task.lua')
	include('Atari_ImageSensor.lua')
	include('AtariWorld_Breakout.lua')

-- Classification
	include('MulticlassClassificationWorld.lua')
	include('MulticlassClassification_Sensor.lua')
	include('MulticlassClassification_Task.lua')

-- Features Acquisition
	include('FeaturesAcquisitionClassificationWorld.lua')
	include('FeaturesAcquisitionClassification_Task.lua')
	include('FeaturesAcquisitionClassification_Sensor.lua')


--include('ClientEnvironment.lua')
--include('ServerEnvironment.lua')

include('Policy.lua')
include('RandomPolicy.lua')
include('PolicyGradient.lua')
include('DeepQPolicy.lua')
include('RecurrentPolicyGradient.lua')
include('StochasticGradientImitationPolicy.lua')
include('PredictiveRecurrentPolicyGradient.lua')

include('ExperimentLog.lua')
include('ExperimentLogCSV.lua')
include('ExperimentLogConsole.lua')
include('ModelsUtils.lua')
include('GRU.lua')
include('RNN.lua')

return rltorch
