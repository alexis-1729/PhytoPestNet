import tensorflow as tf
import yaml
from src.dataset import get_dataset
from src.models import build_model

def train(config_path = "config/cnn_setup.yml"):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


    train_dt, val_dt = get_dataset(config['dataset'])

    model = build_model(config['model']['num_class'])

    model.compile(
        optimizer = tf.keras.optimizers.AdamW(learning_rate = config['training']['learning_rate']),
        loss = tf.keas.losses.SparseCategoricalCrossentropy(),
        metrics =['accuracy']
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath= config['system']['checkpoint_dir'], # gg
        save_best_only = True,
        monitor= "val_loss",
        mode = "min"
    )

    model.fit(
        train_dt,
        validation_data = val_dt,
        epochs = 25,
        callbacks = [checkpoint_callback]
    )

    print("Finalized training")

    if __name__ == "__main__":
        train()