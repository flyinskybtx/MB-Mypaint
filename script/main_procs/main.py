from Model.repr_model import ReprModel
from script.main_procs.collect_policy_data import collect_sim_data
from script.main_procs.hparams import define_hparams
from script.main_procs.make_simulation_environments import sample_env_configs

if __name__ == '__main__':
    # 0. ---------- SETTINGS ----------
    #   0.1 ---------- Define NN h-params ----------
    cfg = define_hparams()

    #   0.2 ---------- Choose Fixed env, random sample other envs -----------
    phy_config = define_hparams()
    sim_configs = sample_env_configs(cfg.num_simulators, True)

    # 1. ---------- TRAIN DYNAMICS  ----------
    #   1.0 ---------- Collect Simulated Data Samples  ----------
    collect_sim_data(sim_configs, cfg.rollout_episodes, remove_old=True)

    #   1.1 ---------- TRAIN Auto-Encoder for image representation  ----------
    repr_model = ReprModel(cfg)

    #   1.2 ---------- TRAIN CNP-dynamics for image prediction  ----------

    #   1.3 ---------- Evaluate Auto-Encoder  ----------
    #   1.4 ---------- Evaluate CNP dynamics  ----------
    #   1.5 ---------- Sample Context Points for dynamics  ----------
    # 2. ---------- Create MPC Controller  ----------
    #   2.0 ---------- Define Trajectory Loss and Create MPC Controller  ----------
    #   2.1 ---------- Evaluate MPC Controller  ----------
    # 3. ---------- Dyna-like Reinforcement learning  ----------
    #   3.0 ---------- Create RL mypaint_painter and surrogate model  ----------
    #   3.1 ---------- Collect MPC Data Samples  ----------
    #   3.1.1 ---------- s-a-s'-r  ----------
    #   3.1.2 ---------- evaluate state value  ----------
    #   3.2 ---------- Supervised Learning for s-a  ----------
    #   3.3 ---------- Supervised Learning for s-r  ----------
    #   3.4 ---------- Vanilla RL Learning  ----------
    # 4. ---------- Collect Data Samples  ----------
    #   4.1 ---------- Collect Data Samples with Exploring Policy ----------
    #   4.2 ---------- Fine-tune Dynamics  ----------
    #   4.3 ---------- Resample Context Points ----------
