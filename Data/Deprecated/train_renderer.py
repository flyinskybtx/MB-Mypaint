import tensorflow.keras as keras

from Data.Deprecated.data_generator import DataGenerator
from Data.Deprecated.renderer import build_generator_model

if __name__ == '__main__':
    model = build_generator_model()
    model.compile(loss='mse')

    train_generator = DataGenerator([f'{1001 + x}-c' for x in range(120)])
    vali_generator = DataGenerator([f'{1121 + x}-c' for x in range(10)], batch_size=1000)
    vali_data = vali_generator.__getitem__(1)

    model.fit_generator(generator=train_generator,
                        validation_data=vali_data,
                        epochs=100, steps_per_epoch=1000,
                        # validation_steps=10,
                        callbacks=[keras.callbacks.ModelCheckpoint('script/checkpoints/checkpoint.{epoch:02d}_{'
                                                                   'val_loss:02f}.hdf5',
                                                                   monitor='val_loss',
                                                                   verbose=0,
                                                                   save_best_only=False, save_weights_only=False,
                                                                   mode='auto', ),

                                   # keras.callbacks.EarlyStopping(monitor='val_loss',
                                   #                               min_delta=0, patience=5, verbose=0,
                                   #                               mode='auto', baseline=None, restore_best_weights=False)
                                   ],
                        )
