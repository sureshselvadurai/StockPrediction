import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



class modelEDA:
    def __init__(self, model, xtrain, symbol, model_features, data):
        self.model = model
        self.x_train = xtrain
        self.symbol = symbol
        self.model_features = model_features
        self.data = data

    def generate_report(self):
        self.gradient_features()

    def gradient_features(self):
        inputs = tf.constant(self.x_train, dtype=tf.float32)

        # Create a TensorFlow GradientTape
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predictions = self.model(inputs)

        gradients = tape.gradient(predictions, inputs)
        feature_importance = tf.reduce_mean(tf.abs(gradients), axis=0)
        feature_importance_np = feature_importance.numpy()
        average_array = np.mean(feature_importance_np, axis=0)

        df_importance = pd.DataFrame({"Feature": self.model_features, "Importance": average_array})
        df_importance = df_importance.sort_values(by="Importance", ascending=False)
        df_importance.to_csv(f"data_processor/EDA/report/{self.symbol}_featureGrad.csv")
