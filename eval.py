import tensorflow as tf
import numpy as np
import yaml
from sklearn.metrics import classification_report
from src.dataset import get_dataset

def evaluate(config_path= "config/cnn_setup.yml"):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    test_dir = "ruta"

    test_dt = get_dataset(config['dataset'])
    model = tf.keras.models.load_model(config['system']['checkpoint_dir'])

    all_preds = []
    all_labels = []

    for images, labels in test_dt:
        predictions = model.predict(images, verbose = 0)

        pred_classes = np.argmax(predictions, axis = 1)

        all_preds.extend(pred_classes)
        all_labels.extend(labels.numpy())

if __name__ == "__main__":
    evaluate()