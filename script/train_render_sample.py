from Data.data_generator import DataGenerator
from Model.renderer import build_generator_model

if __name__ == '__main__':
    model = build_generator_model()
    model.compile(loss='mse')
    train_generator = DataGenerator(['1001-c', '1002-c'])
    vali_generator = DataGenerator(['1003-c'])

    train_sample = train_generator.__getitem__(1)
    print('get item')
    y = train_sample[-1]
    img, disp = train_sample[0]

    model.fit(x=[img, disp], y=y, workers=6, epochs=1000)
