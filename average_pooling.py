# Import Modules
import numpy as np
import tensorflow as tf

# Define input image
image = np.array([[2, 2, 7, 3],
                  [9, 4, 6, 1],
                  [8, 5, 2, 4],
                  [3, 1, 2, 6]])

image = image.reshape(1, 4, 4, 1)

# Define model containing just a single average pooling layer
model = tf.keras.Sequential([
    tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
])

# Generate pooled output
output = model.predict(image)

# Print output image
output = np.squeeze(output)
print(output)