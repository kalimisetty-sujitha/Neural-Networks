import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1) Load MNIST & normalize
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.0

# 2) Tiny single-digit model
model = models.Sequential([
    layers.Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training...")
model.fit(x_train, y_train, epochs=2, batch_size=256, verbose=2)

# 3) Predict 2-digit number + SHOW IMAGE
def predict_two_digit(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (56, 28))  # width=56, height=28

    # Split
    left  = img_resized[:, :28]
    right = img_resized[:, 28:]

    # Preprocess
    left  = left.reshape(1,28,28,1).astype('float32')/255.0
    right = right.reshape(1,28,28,1).astype('float32')/255.0

    # Predict
    dL = model.predict(left)[0].argmax()
    dR = model.predict(right)[0].argmax()

    final = f"{dL}{dR}"

    # SHOW IMAGE
    plt.imshow(img_resized, cmap='gray')
    plt.title("Predicted: " + final)
    plt.axis('off')
    plt.show()

    print("Predicted two-digit number:", final)

# 4) User input
path = input("Enter two-digit image path: ")
predict_two_digit(path)
