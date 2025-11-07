import sys

try:
    import keras
    from keras.models import load_model
except Exception as e:
    print("[ERROR] Keras not available:", e)
    sys.exit(1)

MODEL_PATH = "ConvMixer_WEIGHTS_BIASES.keras"

try:
    m = load_model(MODEL_PATH)
except Exception as e:
    print("[ERROR] Could not load model. If this is weights-only, you'll need the model architecture code to load.")
    print("Exception:", e)
    sys.exit(2)

print("Model loaded.")
print("Inputs:", m.input_shape)
print("Outputs:", m.output_shape)
