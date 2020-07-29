# 渲染器模型
# 输入： 移动量x, y, theta 当前量图片 p
# 输出： 效果量 p 或 效果增量 delta
import tensorflow.keras as keras


def build_generator_model():
    input_image = keras.layers.Input(shape=(33, 33, 1), name='image_input')
    input_disp = keras.layers.Input(shape=(3,), name='displacement_input')

    x = keras.layers.Conv2D(16, (3, 3), 2, name='conv1')(input_image)
    x = keras.layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = keras.layers.Conv2D(32, (3, 3), 1, name='conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.concatenate(inputs=[x, input_disp], name='concat')
    x = keras.layers.Dense(288, name='dense1')(x)
    x = keras.layers.Reshape((3, 3, 32), name='reshape')(x)
    x = keras.layers.UpSampling2D((2, 2), name='depool1')(x)
    x = keras.layers.Conv2DTranspose(32, (3, 3), name='deconv1')(x)
    x = keras.layers.UpSampling2D((2, 2), name='depool2')(x)
    x = keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), name='deconv2')(x)

    model = keras.models.Model(inputs=[input_image, input_disp], outputs=[x])
    model.summary()
    return model

if __name__ == '__main__':
    model = build_generator_model()