from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

_load_env_plugins()
# try 2000
register(id='lorenz_u-v0',entry_point='Lorenz_envs.envs:Lorenzu', max_episode_steps=5000,) 

register(id='rossler-v0',entry_point='Lorenz_envs.envs:Rossler', max_episode_steps=6500,)

register(id='lorenz_y-v0',entry_point='Lorenz_envs.envs:Lorenzy', max_episode_steps=1000,) 

register(id='lorenz_r-v0',entry_point='Lorenz_envs.envs:Lorenzr', max_episode_steps=1000,) 
