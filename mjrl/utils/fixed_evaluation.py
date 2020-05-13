import gym
import pickle
import os

from mjrl.samplers import get_tajectories_per_cpu
from mjrl.utils.gym_env import GymEnv


DEFAULT_MJRL_CACHE_PATH = os.path.expanduser("~/.mjrl")
BASE_SEED = 1234


def generate_missing_init_states(env_name, env_kwargs, existing_states, states_num_to_generate):
    should_generate_and_save = len(existing_states) != states_num_to_generate
    if should_generate_and_save:
        env = gym.make(env_name, **env_kwargs)

        for i in range(len(existing_states), states_num_to_generate):
            env.seed(BASE_SEED + i)
            env.reset()
            existing_states.append(env.get_env_state())

        save_init_states(existing_states, env_name, env_kwargs)

    return existing_states


def get_hash_env_kwargs(env_kwargs):
    key_dict = env_kwargs.copy()
    # 'use_timestamp' key should be ignored, it is used similarly as an algorithm parameter
    # and thus should not influence the test set
    if 'use_timestamp' in key_dict:
        del key_dict['use_timestamp']
    return frozenset(key_dict.items())


def save_init_states(init_states_to_save, env_name, env_kwargs):
    init_states_dict = {}
    env_states_cache_file_path = env_name_to_cache_file_path(env_name)
    if os.path.isfile(env_states_cache_file_path):
        init_states_dict = pickle.load(open(env_states_cache_file_path, 'rb'))
    complete_init_states_to_save = []
    hash_env_kwargs = get_hash_env_kwargs(env_kwargs)
    if hash_env_kwargs in init_states_dict:
        complete_init_states_to_save = init_states_dict[hash_env_kwargs]
    complete_init_states_to_save += init_states_to_save
    init_states_dict[hash_env_kwargs] = complete_init_states_to_save
    os.makedirs(os.path.dirname(env_states_cache_file_path), exist_ok=True)
    pickle.dump(init_states_dict, open(env_states_cache_file_path, 'wb'))


def env_name_to_cache_file_path(env_name):
    mjrl_cache_path = os.environ.get('MJRL_CACHE')
    if mjrl_cache_path is None:
        mjrl_cache_path = DEFAULT_MJRL_CACHE_PATH
    return os.path.join(mjrl_cache_path, env_name + '.pkl')


def read_init_states(env_name, env_kwargs, required_init_state_count):
    loaded_init_states = []
    env_states_cache_file_path = env_name_to_cache_file_path(env_name)
    if os.path.exists(env_states_cache_file_path):
        init_states_dict = pickle.load(open(env_states_cache_file_path, 'rb'))
        hash_env_kwargs = get_hash_env_kwargs(env_kwargs)
        if hash_env_kwargs in init_states_dict:
            loaded_init_states = init_states_dict[hash_env_kwargs]
    loaded_init_states = generate_missing_init_states(env_name, env_kwargs, loaded_init_states,
                                                      required_init_state_count)
    return loaded_init_states


def get_env_name(env):
    if isinstance(env, str):
        env_name = env
    elif isinstance(env, GymEnv):
        env_name = env.env_id
    elif isinstance(env, gym.Env):
        env_name = env.unwrapped.spec.id
    else:
        raise ValueError("Unsupported env variable type", env)
    return env_name


def get_init_states_per_cpu(env, trajectories_number, num_cpu, env_kwargs):
    trajectories_per_cpu = get_tajectories_per_cpu(trajectories_number, num_cpu)
    total_trajectories_number = trajectories_per_cpu * num_cpu

    env_name = get_env_name(env)
    all_init_states = read_init_states(env_name, env_kwargs, total_trajectories_number)
    init_states_per_cpu = []
    i = 0
    for _ in range(num_cpu):
        cpu_init_states = []
        for _ in range(trajectories_per_cpu):
            cpu_init_states.append(all_init_states[i])
            i += 1
        init_states_per_cpu.append(cpu_init_states)
    assert all(len(cpu_states) == len(init_states_per_cpu[0]) for cpu_states in init_states_per_cpu)
    return init_states_per_cpu
