import streamlit as st
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from torch import nn
from PIL import Image
import numpy as np
import glob
import datasets
import matplotlib.pyplot as plt



def distinct_palette():
    return [
        [0, 0, 0],       # Black
        [230, 25, 75],   # Red
        [60, 180, 75],   # Green
        [255, 225, 25],  # Yellow
        [0, 130, 200],   # Blue
        [245, 130, 48],  # Orange
        [145, 30, 180],  # Purple
        [240, 240, 240],  # White
    ]

def get_seg_overlay(image, seg):
  color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
  print(color_seg.shape)
  palette = np.array(distinct_palette())
  for label, color in enumerate(palette):
      color_seg[seg == label, :] = color

  # Show image + mask

  img = np.array(image) * 0.5 + color_seg * 0.5
  img = img.astype(np.uint8)

  return img

def plot_image_result(image, pred_seg, legend_size=5):
  pred_img = get_seg_overlay(image, pred_seg)

  f, axs = plt.subplots()
  img = axs.imshow(pred_img, cmap='viridis')
  #add legend
  legend_labels = ['above ILM','ILM-IPL/INL','IPL/INL-RPE','RPE-BM','under BM','PED','SRF','IRF']
  colors = distinct_palette()
  custom_legend = [plt.Line2D([0], [0], color=np.array(color) / 255, lw=4) for color in colors]
  plt.legend(custom_legend, legend_labels, loc='upper right', prop={'size': legend_size})

  # Render the plot to a NumPy array
  f.canvas.draw()
  image_np = np.frombuffer(f.canvas.tostring_rgb(), dtype=np.uint8)
  image_np = image_np.reshape(f.canvas.get_width_height()[::-1] + (3,))
  plt.close(f)

  return image_np



# Define the model and feature extractor
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained('coralexbadea/Segformer_OCT_Retina')

# Function to perform segmentation
def perform_segmentation(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=(image.size[::-1]),
        mode='bilinear',
        align_corners=False
    )
    pred_seg = upsampled_logits.argmax(dim=1)
    return pred_seg

# Streamlit app
def main():
    st.title("Image Segmentation App")

    uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            seg_result = perform_segmentation(uploaded_file)
            seg_overlay = plot_image_result(Image.open(uploaded_file), seg_result[0].cpu().numpy())
            st.image(seg_overlay, caption='Segmentation Result.', use_column_width=True)

if __name__ == "__main__":
    main()

