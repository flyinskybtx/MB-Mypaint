if __name__ == '__main__':
    pass
    # 0. ---------- SETTINGS ----------
    # 1. ---------- TRAIN DYNAMICS  ----------
    #   1.0 ---------- Collect Data Samples  ----------
    #   1.1 ---------- TRAIN Auto-Encoder for image representation  ----------
    #   1.2 ---------- TRAIN CNP-dynamics for image prediction  ----------
    #   1.3 ---------- Evaluate Auto-Encoder  ----------
    #   1.4 ---------- Evaluate CNP dynamics  ----------
    #   1.5 ---------- Sample Context Points for dynamics  ----------
    # 2. ---------- Create MPC Controller  ----------
    #   2.0 ---------- Define Trajectory Loss and Create MPC Controller  ----------
    #   2.1 ---------- Evaluate MPC Controller  ----------
    # 3. ---------- Dyna-like Reinforcement learning  ----------
    #   3.0 ---------- Create RL agent and surrogate model  ----------
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
