from tensorflow import layers, models

def build_model(num_class = 15, input_shape = (64, 64, 3)):
    
    inputs = layers.Input(shape = input_shape, name = "input_inicial")

    # first layer
    x = layers.Conv2D(32, (3, 3), activation = 'relu', name = "conv2d_1")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    #second layer

    x = layers.Conv2D(32, (3, 3,), activation = 'relu', name = "conv2d_2")(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(num_class, activation = 'softmax', name = "predictions")(x)

    model = models.Model(inputs = inputs, outputs = outputs, name = "phytoNet")

    return model
