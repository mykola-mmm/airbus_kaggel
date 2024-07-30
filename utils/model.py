from tensorflow.keras import layers, Model, Input


def unet(input_shape, optimizer, loss, metrics, gaussian_noise, dropout, num_filters):
    # Inputs
    inputs = Input(input_shape)
    # inputs = Input((None, None, 3))
    inputs = layers.GaussianNoise(gaussian_noise)(inputs)
    inputs = layers.BatchNormalization()(inputs)

    # Encoder
    c1 = layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = layers.Dropout(dropout) (c1)
    c1 = layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = layers.MaxPooling2D((2, 2)) (c1)

    c2 = layers.Conv2D(num_filters*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = layers.Dropout(dropout) (c2)
    c2 = layers.Conv2D(num_filters*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = layers.MaxPooling2D((2, 2)) (c2)

    c3 = layers.Conv2D(num_filters*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = layers.Dropout(dropout) (c3)
    c3 = layers.Conv2D(num_filters*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = layers.MaxPooling2D((2, 2)) (c3)

    c4 = layers.Conv2D(num_filters*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = layers.Dropout(dropout) (c4)
    c4 = layers.Conv2D(num_filters*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = layers.MaxPooling2D((2, 2)) (c4)

    # Bottleneck
    c5 = layers.Conv2D(num_filters*16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = layers.Dropout(dropout) (c5)
    c5 = layers.Conv2D(num_filters*16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

    # Decoder
    u6 = layers.Conv2DTranspose(num_filters*8, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(num_filters*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = layers.Dropout(dropout) (c6)
    c6 = layers.Conv2D(num_filters*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = layers.Conv2DTranspose(num_filters*4, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(num_filters*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = layers.Dropout(dropout) (c7)
    c7 = layers.Conv2D(num_filters*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = layers.Conv2DTranspose(num_filters*2, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(num_filters*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = layers.Dropout(dropout) (c8)
    c8 = layers.Conv2D(num_filters*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = layers.Dropout(dropout) (c9)
    c9 = layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model