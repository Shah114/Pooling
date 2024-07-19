# Importing Modules
import numpy as np
import tensorflow as tf

# Define input image
image = np.array([[2, 2, 7, 3],
                  [9, 4, 6, 1],
                  [8, 5, 2, 4],
                  [3, 1, 2, 6]])

image = image.reshape(1, 4, 4, 1)

# Define gm_model containing just a single global-max pooling layer
gm_model = tf.keras.Sequential([
    tf.keras.layers.GlobalMaxPooling2D()
])
 
# Define ga_model containing just a single global-average pooling layer
ga_model = tf.keras.Sequential([
    tf.keras.layers.GlobalAveragePooling2D()
])
 
# Generate pooled output
gm_output = gm_model.predict(image)
ga_output = ga_model.predict(image)
 
# Print output image
gm_output = np.squeeze(gm_output)
ga_output = np.squeeze(ga_output)
print(f"Global Max output: {gm_output}")
print(f"Global Average output: {ga_output}")