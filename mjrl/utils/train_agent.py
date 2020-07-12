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


def _load_latest_policy_and_logs(agent, policy_dir, logs_dir, should_fresh_start):
    """Loads the latest policy.
    Returns the next step number to begin with.
    """
    if not os.path.isdir(logs_dir):
        agent.global_status['best_perf'] = -1e8
        print("\n--- Warning: no logging activated, the training will be started from scratch. ---\n")
        return 0  # there is no logging activated
    assert os.path.isdir(policy_dir), str(policy_dir)

    log_csv_path = os.path.join(logs_dir, 'log.csv')
    if should_fresh_start:
        assert not os.path.exists(log_csv_path), "Job is already initialized, log should be empty."
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
            agent.load_checkpoint(pickle.load(fp), path=policy_dir)

        return i+1

    # cannot find any saved policy
    raise RuntimeError("Log file exists, but cannot find any saved policy.")


def calculate_policy_update_count(i, irl_kwargs):
    if irl_kwargs is None:
        return 1
    if 'policy_updates' in irl_kwargs:
        return irl_kwargs['policy_updates']
    ratio = i / irl_kwargs['policy']['steps_till_max']
    policy_updates = (1 - ratio) * irl_kwargs['policy']['min_updates'] + ratio * irl_kwargs['policy']['max_updates']
    if policy_updates > irl_kwargs['policy']['max_updates']:
        policy_updates = irl_kwargs['policy']['max_updates']
    if policy_updates < 1:
        policy_updates = 1
    return int(policy_updates)


def check_run_folders(training_path, run_no):
    if not os.path.isdir(training_path):
        os.makedirs(training_path)
    elems_in_training_path = os.listdir(training_path)
    for elem in elems_in_training_path:
        if 'run' not in elem and elem != '.DS_Store' and elem != '.' and elem != '..' and elem != 'config.yaml':
            print('Element in runs path:', elem)
            print('Make sure, that only runs folders are in training path. Exiting.')
            exit(1)
    training_path = os.path.join(training_path, 'run_' + str(run_no))
    if os.path.exists(training_path):
        print("Warning: Run path", training_path, 'already exists.')
    return training_path


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
                plot_keys = None,
                irl_kwargs = None,
                env_kwargs = None,
                temperature_decay=0.95,
                temperature_min=0,
                temperature_max=0,
                training_folder='Runs',
                should_fresh_start=False,
                run_no=None,
                fixed_evaluation_init_states=False
                ):

    np.random.seed(seed)
    print("Job name:", job_name)
    training_path = os.path.join(training_folder, job_name)
    if plot_keys is None:
        plot_keys = ['stoc_pol_mean']
    if run_no is not None:
        training_path = check_run_folders(training_path, run_no)
    if not os.path.isdir(training_path):
        os.makedirs(training_path)
    previous_dir = os.getcwd()
    os.chdir(training_path) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') == False and agent.save_logs == True: os.mkdir('logs')
    best_policy = copy.deepcopy(agent.policy)
    mean_evaluation_pol_performance = 0.0
    if isinstance(env_kwargs, dict):
        e = GymEnv(agent.env.env_id, **env_kwargs)
    else:
        e = GymEnv(agent.env.env_id)

    i_start = _load_latest_policy_and_logs(agent,
                                           policy_dir='iterations',
                                           logs_dir='logs',
                                           should_fresh_start=should_fresh_start)
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
        if run_no is not None:
            print("ITERATION : %i, RUN : %i " % (i, run_no))
        else:
            print("ITERATION : %i " % i)

        new_temperature = (temperature_max-temperature_min)*(temperature_decay ** i) + temperature_min
        if new_temperature < 0 or temperature_max == 0:
            new_temperature = 0
        agent.policy.set_temperature(new_temperature)
        if agent.save_logs:
            agent.logger.log_kv('temperature', new_temperature)
        if train_curve[i-1] > agent.global_status['best_perf']:
            best_policy = copy.deepcopy(agent.policy)
            agent.global_status['best_perf'] = train_curve[i-1]

        N = num_traj if sample_mode == 'trajectories' else num_samples
        args = dict(N=N, itr=i, sample_mode=sample_mode, gamma=gamma, gae_lambda=gae_lambda, num_cpu=num_cpu,
                    env_kwargs=env_kwargs)
        # calculate no. of policy updates (used for IRL)
        policy_updates_count = calculate_policy_update_count(i, irl_kwargs)
        if irl_kwargs is not None:
            args['return_paths'] = True
        sampler_paths = []
        # do policy update
        for j in range(policy_updates_count):
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

        # IRL discriminator update
        if irl_kwargs is not None:
            agent.fit_irl(sampler_paths, main_loop_step=i, main_loop_percentage=i/niter,
                          num_cpu=num_cpu, policy_updates_count=policy_updates_count)

        if evaluation_rollouts is not None and evaluation_rollouts > 0:
            print("Performing evaluation rollouts ........")
            eval_paths = sample_paths(num_traj=evaluation_rollouts, policy=agent.policy, num_cpu=num_cpu,
                                      env=e.env_id, eval_mode=True, base_seed=seed, env_kwargs=env_kwargs,
                                      fixed_init_states=fixed_evaluation_init_states)
            if hasattr(agent, "irl_model"):
                eval_paths = agent.eval_irl(eval_paths, training_paths_from_policy=False)
            mean_evaluation_pol_performance = np.mean([np.sum(path['rewards']) for path in eval_paths])
            if agent.save_logs:
                agent.logger.log_kv('eval_score', mean_evaluation_pol_performance)
                eval_success_rate = e.env.evaluate_success(eval_paths)
                agent.logger.log_kv('eval_success_rate', eval_success_rate)

        if agent.save_logs:
            agent.logger.align_rows()

        if i % save_freq == 0 and i > 0:
            save_progress()

        # print results to console
        if i == 0:
            result_file = open('results.txt', 'w')
            print("Iter | Stoc Pol | Mean Pol | Best (Stoc) \n")
            result_file.write("Iter | Sampling Pol | Evaluation Pol | Best (Sampled) \n")
            result_file.close()
        print("[ %s ] %4i %5.2f %5.2f %5.2f " % (timer.asctime(timer.localtime(timer.time())),
                                                 i, train_curve[i], mean_evaluation_pol_performance, agent.global_status['best_perf']))
        result_file = open('results.txt', 'a')
        result_file.write("%4i %5.2f %5.2f %5.2f \n" % (i, train_curve[i], mean_evaluation_pol_performance, agent.global_status['best_perf']))
        result_file.close()
        if agent.save_logs:
            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                       agent.logger.get_current_log().items()))
            print(tabulate(print_data))

    # final save
    if i_start < niter:
        save_progress()
    else:
        print("Requested iteration number equal to the found checkpoint iteration count. All done, exiting.")

    os.chdir(previous_dir)
