from Model.cnp_model import CNP
from script.main_procs.hparams import define_hparams

if __name__ == '__main__':
    cfg = define_hparams()
    dynamics_model = CNP(cfg)

    dynamics_model.build_graph(input_shape=[(None, None, 22), (None, None, 7), (None, 22)])

    dynamics_model.summary()
