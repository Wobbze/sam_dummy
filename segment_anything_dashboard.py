import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
st.set_page_config(page_title="Segment Anything App", layout="wide")

import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from huggingface_hub import hf_hub_download
from streamlit_image_coordinates import streamlit_image_coordinates
import io

# Load SAM model
@st.cache_resource
def load_sam_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = hf_hub_download(
        repo_id="ybelkada/segment-anything",
        filename="checkpoints/sam_vit_b_01ec64.pth"
    )
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor

predictor = load_sam_model()

# App UI
st.title("🧠 Segment Anything with Click")

uploaded_file = st.sidebar.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image).astype(np.float32)
    predictor.set_image(image_np)

    st.subheader("🖱️ Click a point on the image to segment")
    coords = streamlit_image_coordinates(image, key="click")

    if coords:
        x, y = int(coords["x"]), int(coords["y"])
        st.success(f"Clicked at: ({x}, {y})")

        input_point = np.array([[x, y]])
        input_label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        best_mask = masks[np.argmax(scores)]
        segmented_image = image_np.copy()
        segmented_image[~best_mask] = 0

        col1, col2 = st.columns(2)
        col1.image(best_mask, caption="Segmentation Mask", use_column_width=True)
        col2.image(segmented_image, caption="Segmented Output", use_column_width=True)

        # Download
        seg_pil = Image.fromarray(segmented_image)
        buf = io.BytesIO()
        seg_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button("💾 Download Segmented Image", data=byte_im, file_name="segmented.png", mime="image/png")
    else:
        st.info("Click on the image above to generate a mask.")
else:
    st.info("👈 Upload an image to get started.")
