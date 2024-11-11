import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import cv2
from processing_func import grayscaling, black_and_white, negative

#------------------ STYLING --------------------#
st.markdown(
  """
  <style>
  .center{
    text-align: center;
  }
  </style>
  """,
  unsafe_allow_html=True
)
st.markdown("<h2 class='center'>Website Pengolahan Citra</h2>", unsafe_allow_html=True)

# input gambar
image_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

option = 'default'

#------------------ DISPLAY GAMBAR, 2 KOLOM --------------------#
# Bagi ke 2 kolom. kolom 1 gambar asli, kolom 2 gambar proses
def display():
  col1, col2 = st.columns(2)

  with col1:
    st.markdown("<h3 class='center'>Gambar asli</h3>", unsafe_allow_html=True)
    if image_file is not None:
      st.image(image_file, use_column_width=True)

      #-------- HISTOGRAM --------#
      image_for_hist = plt.imread(image_file)
      # blue_color = cv2.calcHist([image_for_hist], [0], None, [256], [0, 256]) 
      # red_color = cv2.calcHist([image_for_hist], [1], None, [256], [0, 256]) 
      # green_color = cv2.calcHist([image_for_hist], [2], None, [256], [0, 256]) 

      # fig, ax = plt.subplots(figsize=(4, 1))
      # ax.set_title("Histogram")

      # # Plot histograms for each color channel
      # ax.hist(blue_color, bins=256, color="blue", alpha=0.5, label="Blue")
      # ax.hist(green_color, bins=256, color="green", alpha=0.5, label="Green")
      # ax.hist(red_color, bins=256, color="red", alpha=0.5, label="Red")

      # st.pyplot(fig)

      imgRGB = cv2.cvtColor(image_for_hist, cv2.COLOR_BGR2RGB)

      plt.figure(figsize=(4, 2))
      colors = ['b', 'g', 'r']
      for i in range(len(colors)):
        hist = cv2.calcHist([imgRGB], [i], None, [256], [0, 256])
        plt.plot(hist, colors[i])
      plt.title('Histogram')

      st.pyplot(plt)

  # Col 2 image hasil
  with col2:
    st.markdown("<h3 class='center'>Hasil Proses</h3>", unsafe_allow_html=True)
    
    if image_file is not None:
      image_process = plt.imread(image_file) # convert to numpy array
      
      if option=='default':
        st.image(image_file, use_column_width=True)
      elif option=='grayscale':
        gray_img = grayscaling(image_process)

        fig, ax = plt.subplots()
        ax.imshow(gray_img, cmap='grey')
        ax.axis('off')  # Hide axes
        st.pyplot(fig)
      elif option=='black and white':

        gray_img = black_and_white(image_process)

        fig, ax = plt.subplots()
        ax.imshow(gray_img, cmap='grey')
        ax.axis('off')  # Hide axes
        st.pyplot(fig)
      elif option=='negative':
        gray_img = negative(image_process)

        fig, ax = plt.subplots()
        ax.imshow(gray_img, cmap='grey')
        ax.axis('off')  # Hide axes
        st.pyplot(fig)
    
option = st.selectbox(
  'Silahkan pilih menu',
  ('default', 'grayscale', 'black and white', 'negative'),
)

display()



