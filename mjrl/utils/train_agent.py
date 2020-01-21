import logging
logging.disable(logging.CRITICAL)

from tabulate import tabulate
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.utils.gym_env import GymEnv
from mjrl.samplers.core import sample_paths
import numpy as np
import pickle
import time as timer
import os
import copy


def _load_latest_policy_and_logs(agent, policy_dir, logs_dir):
    """Loads the latest policy.
    Returns the next step number to begin with.
    """
    assert os.path.isdir(policy_dir), str(policy_dir)
    assert os.path.isdir(logs_dir), str(logs_dir)

    log_csv_path = os.path.join(logs_dir, 'log.csv')
    if not os.path.exists(log_csv_path):
        agent.global_status['best_perf'] = -1e8
        return 0   # fresh start

    print("Reading: {}".format(log_csv_path))
    agent.logger.read_log(log_csv_path)
    last_step = agent.logger.max_len - 1

    # find latest policy/baseline
    i = last_step
    while i >= 0:
        checkpoint_path = os.path.join(policy_dir, 'checkpoint_{}.pickle'.format(i))
        if not os.path.isfile(checkpoint_path):
            i -= 1
            continue

        with open(checkpoint_path, 'rb') as fp:
            agent.load_checkpoint(pickle.load(fp), path=policy_dir, iteration=i)

        return i+1

    # cannot find any saved policy
    raise RuntimeError("Log file exists, but cannot find any saved policy.")


def calculate_policy_update_count(i, irl_kwargs):
    if irl_kwargs is None:
        return 1
    if 'policy_updates' in irl_kwargs:
        return irl_kwargs['policy_updates']
    policy_updates = irl_kwargs['policy']['min_updates'] + irl_kwargs['policy']['max_updates'] * \
                     (i / irl_kwargs['policy']['steps_till_max'])
    if policy_updates > irl_kwargs['policy']['max_updates']:
        policy_updates = irl_kwargs['policy']['max_updates']
    if policy_updates < 1:
        policy_updates = 1
    if int(policy_updates) == 3:
        policy_updates = 4
    return int(policy_updates)


def train_agent(job_name, agent,
                seed = 0,
                niter = 101,
                gamma = 0.995,
                gae_lambda = None,
                num_cpu = 1,
                sample_mode = 'trajectories',
                num_traj = 50,
                num_samples = 50000, # has precedence, used with sample_mode = 'samples'
                save_freq = 10,
                evaluation_rollouts = None,
                plot_keys = ['stoc_pol_mean'],
                irl_kwargs = None,
                env_kwargs = None,
                temperature_base=0,
                temperature_decay=0.95
                ):

    np.random.seed(seed)
    if os.path.isdir(job_name) == False:
        os.mkdir(job_name)
    previous_dir = os.getcwd()
    os.chdir(job_name) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') == False and agent.save_logs == True: os.mkdir('logs')
    best_policy = copy.deepcopy(agent.policy)
    mean_pol_perf = 0.0
    if isinstance(env_kwargs, dict):
        e = GymEnv(agent.env.env_id, **env_kwargs)
    else:
        e = GymEnv(agent.env.env_id)

    i_start = _load_latest_policy_and_logs(agent,
                                           policy_dir='iterations',
                                           logs_dir='logs')
    train_curve = agent.global_status['best_perf'] * np.ones(niter)

    def save_progress():
        if agent.save_logs:
            agent.logger.save_log('logs/')
            make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
        checkpoint_file = 'checkpoint_%i.pickle' % i
        pickle.dump(agent.checkpoint, open('iterations/' + checkpoint_file, 'wb'))
        # check if agent has custom save_checkpoint function defined, if so use it
        save_checkpoint_funct = getattr(agent, "save_checkpoint", None)
        if save_checkpoint_funct:
            save_checkpoint_funct(path='iterations/', iteration=i)
        pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))

    if i_start:
        print("Resuming from an existing job folder ...")


    for i in range(i_start, niter):
        print("......................................................................................")
        print("ITERATION : %i " % i)

        new_temperature = temperature_base * (temperature_decay ** i)
        agent.policy.set_temperature(new_temperature)
        agent.logger.log_kv('temperature', new_temperature)
        if train_curve[i-1] > agent.global_status['best_perf']:
            best_policy = copy.deepcopy(agent.policy)
            agent.global_status['best_perf'] = train_curve[i-1]

        N = num_traj if sample_mode == 'trajectories' else num_samples
        args = dict(N=N, sample_mode=sample_mode, gamma=gamma, gae_lambda=gae_lambda, num_cpu=num_cpu,
                    env_kwargs=env_kwargs)
        # calculate no. of policy updates (used for IRL)
        policy_updates_no = calculate_policy_update_count(i, irl_kwargs)
        if irl_kwargs is not None:
            args['return_paths'] = True
            sampler_paths = []
        # do policy update
        for j in range(policy_updates_no):
            output = agent.train_step(**args)
            if isinstance(output, tuple):
                sampler_paths.extend(output[1])
                stats = output[0]
            else:
                stats = output
            if j == 0:
                train_curve[i] = stats[0]
            else:
                train_curve[i] = train_curve[i] + (1/(1+j)*(stats[0] - train_curve[i]))

        if agent.save_logs:
            agent.logger.log_kv('iteration', i)
        agent.logger.align_rows()

        # IRL discriminator update
        if irl_kwargs is not None:
            agent.fit_irl(sampler_paths)

        if evaluation_rollouts is not None and evaluation_rollouts > 0:
            print("Performing evaluation rollouts ........")
            eval_paths = sample_paths(num_traj=evaluation_rollouts, policy=agent.policy, num_cpu=num_cpu,
                                      env=e.env_id, eval_mode=True, base_seed=seed)
            mean_pol_perf = np.mean([np.sum(path['rewards']) for path in eval_paths])
            if agent.save_logs:
                agent.logger.log_kv('eval_score', mean_pol_perf)

        if i % save_freq == 0 and i > 0:
            save_progress()

        # print results to console
        if i == 0:
            result_file = open('results.txt', 'w')
            print("Iter | Stoc Pol | Mean Pol | Best (Stoc) \n")
            result_file.write("Iter | Sampling Pol | Evaluation Pol | Best (Sampled) \n")
            result_file.close()
        print("[ %s ] %4i %5.2f %5.2f %5.2f " % (timer.asctime(timer.localtime(timer.time())),
                                                 i, train_curve[i], mean_pol_perf, agent.global_status['best_perf']))
        result_file = open('results.txt', 'a')
        result_file.write("%4i %5.2f %5.2f %5.2f \n" % (i, train_curve[i], mean_pol_perf, agent.global_status['best_perf']))
        result_file.close()
        if agent.save_logs:
            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                       agent.logger.get_current_log().items()))
            print(tabulate(print_data))

    # final save
    if i_start != niter:
        save_progress()
    else:
        print("Requested iteration number equal to the found checkpoint iteration count. All done, exiting.")

    os.chdir(previous_dir)
