import tensorflow as tf

def get_dataset(data_config):

    train_dt = tf.keras.utils.image_dataset_from_directory(
        data_config['path'],
        validation_split = 0.2,
        subset = "training",
        seed = 123, 
        image_size = (data_config['img_height'], data_config['img_width']),
        batch_size = data_config['batch_size'] 
    )

    val_dt = tf.keras.utils.image_dataset_from_directory(
        data_config['path'],
        validation_split = 0.2,
        subset = "validation",
        seed = 123,
        image_size = (data_config['img_height'], data_config['img_width']),
        batch_size = data_config['batch_size'] 
    )

    data_augmentetion = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1)
    ])

    train_dt = train_dt.map(lambda x, y: (data_augmentetion(x, training = True), y))

    train_dt = train_dt.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    val_dt = val_dt.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

    return train_dt, val_dt
