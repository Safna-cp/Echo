import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
import matplotlib.pyplot as plt

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Echocardiogram ViT + XAI",
    layout="centered"
)

IMG_SIZE = 224

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ProposedViT_best.keras")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")

# ============================================================
# CUSTOM LAYERS (REQUIRED FOR MODEL LOAD)
# ============================================================
class Patches(L.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        return tf.reshape(patches, [tf.shape(images)[0], -1, patches.shape[-1]])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg


class PatchEncoder(L.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.projection = L.Dense(projection_dim)
        self.position_embedding = L.Embedding(num_patches, projection_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=x.shape[1], delta=1)
        return self.projection(x) + self.position_embedding(positions)

    def get_config(self):
        return super().get_config()

# ============================================================
# LOAD MODEL + LABELS (CACHED)
# ============================================================
@st.cache_resource
def load_model_and_labels():
    with open(LABEL_MAP_PATH) as f:
        lm = json.load(f)

    class_names = [
        lm["idx_to_class"][str(i)]
        for i in range(len(lm["idx_to_class"]))
    ]

    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder},
    )

    # Transformer token output layer
    token_layer = model.get_layer("tokens_ln_final")
    token_model = keras.Model(model.input, token_layer.output)

    return model, token_model, class_names


model, token_model, CLASS_NAMES = load_model_and_labels()

# ============================================================
# IMAGE UTILITIES
# ============================================================
def preprocess(img):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img[None, ...]

def vit_token_cam(img_tensor):
    """
    Transformer-friendly XAI:
    Token-norm CAM (robust for ViT)
    """
    tokens = token_model(img_tensor, training=False)[0].numpy()  # (N, D)
    norms = np.linalg.norm(tokens, axis=-1)                      # (N,)
    side = int(np.sqrt(len(norms)))

    cam = norms.reshape(side, side)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE)).numpy()[..., 0]
    return cam

def overlay_cam(img, cam, alpha=0.35):
    import matplotlib.cm as cm
    heatmap = cm.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * img + alpha * heatmap, 0, 1)

# ============================================================
# STREAMLIT UI
# ============================================================
st.title("🫀 Echocardiogram ViT Classification with XAI")
st.write(
    "Upload an echocardiogram image to get a prediction and "
    "a **Vision Transformer explanation (Token-based CAM)**."
)

uploaded = st.file_uploader(
    "Upload image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    # Load image
    img = tf.io.decode_image(uploaded.read(), channels=3)
    img_np = img.numpy()

    st.subheader("Input Image")
    st.image(img_np, use_container_width=True)

    # Predict
    x = preprocess(img)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    st.subheader("Prediction")
    st.markdown(
        f"""
        **Class:** `{CLASS_NAMES[pred_idx]}`  
        **Confidence:** `{probs[pred_idx] * 100:.2f}%`
        """
    )

    st.subheader("Class Probabilities")
    for i, cls in enumerate(CLASS_NAMES):
        st.progress(float(probs[i]), text=f"{cls}: {probs[i]*100:.2f}%")

    # ========================================================
    # XAI: ViT Token CAM
    # ========================================================
    cam = vit_token_cam(x)
    overlay = overlay_cam(img_np / 255.0, cam)

    st.subheader("🧠 XAI – ViT Token Attribution")

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Token Importance Map")
        fig1, ax1 = plt.subplots()
        ax1.imshow(cam, cmap="jet")
        ax1.axis("off")
        st.pyplot(fig1)

    with col2:
        st.caption("Overlay on Image")
        fig2, ax2 = plt.subplots()
        ax2.imshow(overlay)
        ax2.axis("off")
        st.pyplot(fig2)

    st.info(
        "This explanation uses **Transformer token-norm attribution**, "
        "a stable XAI method for Vision Transformers (ViT)."
    )
