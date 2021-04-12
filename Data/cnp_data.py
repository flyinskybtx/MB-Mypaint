import numpy as np
from tqdm import tqdm

from Data import BaseData
from Data.Deprecated.repr_model import ReprModel
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
            mu_0, sigma_0 = self.repr_model.latent_encode(batch['obs'])
            mu_1, sigma_1 = self.repr_model.latent_encode(batch['new_obs'])
            embedded_actions = self.action_embed.transform(batch['actions'])
        except AttributeError:
            raise AttributeError
        states = np.concatenate([mu_0, embedded_actions], axis=-1)
        mu = mu_1 - mu_0
        sigma = sigma_1 + sigma_0

        if self.train:
            context_x = np.repeat(np.expand_dims(states[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            context_y = np.repeat(np.expand_dims(mu[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            query_x = states[-self.batch_size:]  # If train, include context points
            mu = mu[-self.batch_size:]
            sigma = sigma[-self.batch_size:]
        else:
            context_x = np.repeat(np.expand_dims(states[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            context_y = np.repeat(np.expand_dims(mu[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            query_x = states[-self.batch_size:]  # If test, select other points as query
            mu = mu[-self.batch_size:]
            sigma = sigma[-self.batch_size:]

            assert context_x.shape[0] == context_y.shape[0] == query_x.shape[0] == mu.shape[0], \
                f'{context_x.shape[0]} {context_y.shape[0]} {query_x.shape[0]} {mu.shape[0]}'

        return {'context_x': context_x, 'context_y': context_y, 'query_x': query_x}, \
               {'mu': mu, 'sigma': sigma}

    def __len__(self):
        min_len = np.min([len(v['actions']) for v in self.data.values()])
        if self.train:
            return min_len // max(self.max_num_context, self.batch_size) - 1
        else:
            return min_len // (self.max_num_context + self.batch_size) - 1

    def on_epoch_end(self):
        self.num_context = np.random.randint(self.min_num_context, self.max_num_context)
        self.shuffle()


class CNPData(BaseData):
    def __init__(self, savedir, batch_size, num_context: tuple, encoder=None, embedder=None, **kwargs):
        super().__init__(savedir, batch_size, **kwargs)
        self.min_num_context, self.max_num_context = num_context
        self.train = kwargs.setdefault('train', False)
        self.encoder = encoder
        self.embedder = embedder

        self.on_epoch_end()

    def _get_batch(self, i, batch_size):
        key = self.data.keys()[0]
        value = self.data[key]
        start = i*batch_size % len(value)
        batch = value[start:start+batch_size]
        if len(batch)<batch_size:
            batch += batch[:batch_size-len(batch)]

    def __getitem__(self, index):
        if self.train:
            batch = self._get_batch(index, max(self.batch_size, self.num_context))
        else:
            batch = self._get_batch(index, self.batch_size + self.num_context)

        # get latent
        obs = batch['obs']
        new_obs = batch['new_obs']
        actions = batch['actions']

        context_obs = np.repeat(np.expand_dims(obs[:self.num_context], axis=0),
                                self.batch_size, axis=0)  # b, n, 64, 64, 4
        context_new_obs = np.repeat(np.expand_dims(new_obs[:self.num_context], axis=0),
                                    self.batch_size, axis=0)  # b, n, 64, 64, 4
        context_actions = np.repeat(np.expand_dims(actions[:self.num_context], axis=0),
                                    self.batch_size, axis=0)  # b, n, 3
        query_obs = obs[-self.batch_size:]
        query_actions = actions[-self.batch_size:]  # b, 3
        target_new_obs = new_obs[-self.batch_size:]  # b, 64, 64, 4
        assert context_obs.shape[0] == target_new_obs.shape[0] > 0

        if self.encoder:
            mu0, _ = self.encoder.predict(context_obs)
            mu1, _ = self.encoder.predict(context_new_obs)
            actions = self.embedder.predict(context_actions)
            context_x = np.concatenate([mu1, actions], axis=-1)
            context_y = mu1 - mu0

            mu0, _ = self.encoder.predict(query_obs)
            mu1, _ = self.encoder.predict(target_new_obs)
            actions = self.embedder.predict(query_actions)
            query_x = np.concatenate([mu1, actions], axis=-1)
            target_y = mu1 - mu0

            return {'context_x': context_x,
                    'context_y': context_y,
                    'query_x': query_x}, target_y
        else:
            return {'context_obs': context_obs,
                    'context_new_obs': context_new_obs,
                    'context_actions': context_actions,
                    'query_obs': query_obs,
                    'query_actions': query_actions}, {'target_new_obs': target_new_obs}

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
    # --- load obs_encoder
    cfg = define_hparams()
    cfg.train_latent_encoder = False
    cfg.train_decoder = False
    cfg.num_context = (10, 20)

    train_data = CNPData(savedir='offline/random',
                         batch_size=32,
                         num_context=cfg.num_context,
                         train=True)

    for i in range(100):
        train_data.on_epoch_end()
        for j in tqdm(range(train_data.__len__())):
            X, Y = train_data.__getitem__(j)
        print(i)
