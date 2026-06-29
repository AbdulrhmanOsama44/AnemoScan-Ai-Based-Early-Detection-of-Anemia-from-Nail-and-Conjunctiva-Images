import cv2
import numpy as np
import tensorflow as tf

def get_gradcam_heatmap(model, img_array, last_conv_layer_name = 'Conv_1'):
    
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
            
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis = (0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()

def overlay_heatmap_masked(img_resized, heatmap, alpha = 0.5, threshold = 10):
    
    # Foreground mask
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = mask / 255.0

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # Apply mask (background becomes black)
    for c in range(3):
        heatmap_colored[:, :, c] = (heatmap_colored[:, :, c] * mask).astype(np.uint8)

    # Overlay
    superimposed = cv2.addWeighted(img_resized, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed