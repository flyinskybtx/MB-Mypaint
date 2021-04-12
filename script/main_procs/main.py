from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import random

from Data.Deprecated.dynamics_model import CNPModel, make_train_data
from Model.action_embedder import ActionEmbedder
from Data.Deprecated.repr_model import ReprModel
from Data.Deprecated.collect_policy_data import collect_sim_data
from Data.Deprecated.evaluate_auto_encoder import evaluate_repr
from Data.Deprecated.evaluate_cnp_dynamics import evaluate_dynamics
from script.main_procs.hparams import define_hparams
from Data.Deprecated.make_simulation_environments import sample_env_configs
from Data.Deprecated.sample_context import gen_context

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
    embedder = ActionEmbedder(cfg)

    #   1.2 ---------- TRAIN CNP-dynamics for image prediction  ----------
    dynamics = CNPModel(cfg)
    if cfg.train_dynamics:
        dynamics.set_repr(repr_model, embedder)
        train_data, vali_data = make_train_data(cfg, repr_model, embedder)
        dynamics.train_model(train_data, vali_data)
        dynamics.save_model()
    else:
        dynamics.load_model()

    #   1.3 ---------- Evaluate Auto-Encoder  ----------
    if cfg.evaluate_vae:
        evaluate_repr(cfg, repr_model)
    #   1.5 ---------- Sample Context Points for dynamics  ----------
    context_x, context_y = gen_context(cfg, repr_model, embedder, num_context_points=random.randint(*cfg.num_context))

    #   1.4 ---------- Evaluate CNP dynamics  ----------
    if cfg.evaluate_dynamics:
        evaluate_dynamics(cfg, dynamics, repr_model, embedder, hijack=5)


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
