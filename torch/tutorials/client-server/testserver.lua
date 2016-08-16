require 'rltorch'
package.path = '/home/denoyer/alewrap/?/init.lua;' .. package.path
local alewrap = require 'alewrap'

env = rltorch.MountainCar_v0()
env = rltorch.Atari_Breakout_v0({rom="environments/ale/roms/breakout.bin"})

--- Use the "torch" version for fast data exchange between torch server and torch client, use "json" if one wants to use a exchange format that is compatible with other platforms...
env=rltorch.ServerEnvironment(env,55555,"torch",true,{mode="qt",fps="30"})
--env=rltorch.ServerEnvironment(env,55555,"json",true,{mode="qt",fps="30"})

 
