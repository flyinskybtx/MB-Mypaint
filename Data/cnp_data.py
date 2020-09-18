import numpy as np

from Data.data_utils import BaseData
from Model.action_embedder import ActionEmbedder
from Model.repr_model import ReprModel
from script.main_procs.hparams import define_hparams


class CNPGenerator(BaseData):
    def __init__(self, savedir, batch_size, repr_model: ReprModel, action_embed, num_context: tuple, **kwargs):
        super().__init__(savedir, batch_size, **kwargs)
        self.repr_model = repr_model
        self.action_embed = action_embed
        self.min_num_context, self.max_num_context = num_context
        self.train = kwargs.setdefault('train', False)
        self.max_samples = kwargs.setdefault('max_samples', None)

        if self.max_samples is not None:
            for k, v in self.data:
                if len(v) > self.max_samples:
                    self.data[k] = v[:int(self.max_samples)]
        self.on_epoch_end()

    def __getitem__(self, index):
        if self.train:
            batch = self._get_batch(index, max(self.batch_size, self.num_context))
        else:
            batch = self._get_batch(index, self.batch_size + self.num_context)

        # get latent
        try:
            cur_latents = self.repr_model.latent_encode(batch['obs'])
            new_latents = self.repr_model.latent_encode(batch['new_obs'])
            embedded_actions = self.action_embed.transform(batch['actions'])
        except AttributeError:
            raise AttributeError
        states = np.concatenate([cur_latents, embedded_actions], axis=-1)
        delta = new_latents - cur_latents

        if self.train:
            context_x = np.repeat(np.expand_dims(states[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            context_y = np.repeat(np.expand_dims(delta[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            query_x = states[-self.batch_size:]  # If train, include context points
            target_y = delta[-self.batch_size:]
        else:
            context_x = np.repeat(np.expand_dims(states[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            context_y = np.repeat(np.expand_dims(delta[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            query_x = states[-self.batch_size:]  # If test, select other points as query
            target_y = delta[-self.batch_size:]

            assert context_x.shape[0] == context_y.shape[0] == query_x.shape[0] == target_y.shape[0], \
                f'{context_x.shape[0]} {context_y.shape[0]} {query_x.shape[0]} {target_y.shape[0]}'

        return {'context_x': context_x, 'context_y': context_y, 'query_x': query_x}, target_y

    def __len__(self):
        min_len = np.min([len(v['actions']) for v in self.data.values()])
        if self.train:
            return min_len // max(self.max_num_context, self.batch_size) - 1
        else:
            return min_len // (self.max_num_context + self.batch_size) - 1

    def on_epoch_end(self):
        self.num_context = np.random.randint(self.min_num_context, self.max_num_context)
        self.shuffle()


if __name__ == '__main__':
    # --- load encoder
    cfg = define_hparams()
    cfg.train_latent_encoder = False
    cfg.train_decoder = False
    cfg.num_context = (10, 20)
    repr_model = ReprModel(cfg)
    action_embed = ActionEmbedder(cfg)

    train_data = CNPGenerator(repr_model=repr_model,
                              action_embed=action_embed,
                              savedir='offline/random',
                              batch_size=16,
                              num_context=cfg.num_context,
                              train=True)

    for i in range(100):
        train_data.on_epoch_end()
        for j in range(train_data.__len__()):
            X, Y = train_data.__getitem__(j)
        print(i)
