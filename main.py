import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import cv2
from PIL import Image
import time

# --------------- FUNCTIONS --------------- #
def plot_histogram_image(image, fig=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(4, 2))
    else:
        fig.clf()
        ax = fig.add_subplot(111)
        
    if len(image.shape) == 2:  # Grayscale
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='black')
    else:  # Color (BGR)
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)
    ax.set_title('Histogram')
    plt.tight_layout()
    return fig

def plot_histogram_video(image, fig=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(4, 2))
    else:
        fig.clf()
        ax = fig.add_subplot(111)
        
    if len(image.shape) == 2:  # Grayscale
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='black')
    else:  # Color (BGR)
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)
    ax.set_title('Histogram')
    plt.tight_layout()
    return fig

# -------------------- Filters -------------------- #
def grayscaling(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def black_and_white(image, threshold=127):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return bw_image

def negative(image):
    return cv2.bitwise_not(image)

def adjust_brightness(image, value=50):
    return cv2.convertScaleAbs(image, alpha=1, beta=value)

def adjust_contrast(image, value=1.5):
    return cv2.convertScaleAbs(image, alpha=value, beta=0)

def gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def edge_detection(image, threshold1=100, threshold2=200):
    return cv2.Canny(image, threshold1, threshold2)

def sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    return cv2.transform(image, kernel)

def emboss(image):
    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])
    return cv2.filter2D(image, -1, kernel)

def adjust_rgb(image, r_value=1.0, g_value=1.0, b_value=1.0):
    b, g, r = cv2.split(image)
    b = cv2.convertScaleAbs(b, alpha=b_value)
    g = cv2.convertScaleAbs(g, alpha=g_value)
    r = cv2.convertScaleAbs(r, alpha=r_value)
    return cv2.merge((b, g, r))

# --------------- Dynamic Filter Application --------------- #
def apply_filters(image, filters):
    for filter_option in filters:
        if filter_option['name'] == 'grayscale':
            image = grayscaling(image)
        elif filter_option['name'] == 'black and white':
            image = black_and_white(image, filter_option['threshold'])
        elif filter_option['name'] == 'negative':
            image = negative(image)
        elif filter_option['name'] == 'brightness':
            image = adjust_brightness(image, filter_option['value'])
        elif filter_option['name'] == 'contrast':
            image = adjust_contrast(image, filter_option['value'])
        elif filter_option['name'] == 'gaussian blur':
            image = gaussian_blur(image, filter_option['kernel_size'])
        elif filter_option['name'] == 'edge detection':
            image = edge_detection(image, filter_option['threshold1'], filter_option['threshold2'])
        elif filter_option['name'] == 'sepia':
            image = sepia(image)
        elif filter_option['name'] == 'emboss':
            image = emboss(image)
        elif filter_option['name'] == 'adjust rgb':
            image = adjust_rgb(image, filter_option['r'], filter_option['g'], filter_option['b'])
    return image

# ------------------ PAGE CONFIGURATION --------------------#
st.set_page_config(layout="wide")

# ------------------ CUSTOM CSS --------------------#
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 10rem;
        padding-right: 10rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h2 class='center'>Image Processing Website</h2>", unsafe_allow_html=True)
tabs = st.tabs(["Image Upload", "Video Input"])

# ------------------ TAB 1: IMAGE UPLOAD --------------------#
# ------------------ TAB 1: IMAGE UPLOAD --------------------#
with tabs[0]:
    st.subheader("Upload and Apply Multiple Filters")

    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # Initialize filters list if not already initialized
    if 'filters' not in st.session_state:
        st.session_state.filters = []

    # Add or Remove Filters
    if st.button("Add Filter"):
        st.session_state.filters.append({'name': 'default'})
        st.rerun(scope="app")  # Refresh the page after adding a filter

    if len(st.session_state.filters) > 1 and st.button("Remove Filter"):
        st.session_state.filters.pop()
        st.rerun(scope="app")  # Refresh the page after removing a filter

    # Display filter options dynamically
    for i, filter_option in enumerate(st.session_state.filters):
        filter_name = st.selectbox(f"Select Filter {i+1}", 
                                   ['default', 'grayscale', 'black and white', 'negative',
                                    'brightness', 'contrast', 'gaussian blur', 'edge detection',
                                    'sepia', 'emboss', 'adjust rgb'],
                                   key=f"filter_select_{i}")
        st.session_state.filters[i]['name'] = filter_name

        # Add sliders based on filter type
        if filter_name == 'brightness':
            st.session_state.filters[i]['value'] = st.slider(f'Brightness Level {i+1}', -100, 100, 50, key=f"brightness_{i}")
        elif filter_name == 'contrast':
            st.session_state.filters[i]['value'] = st.slider(f'Contrast Level {i+1}', 0.1, 3.0, 1.5, key=f"contrast_{i}")
        elif filter_name == 'gaussian blur':
            st.session_state.filters[i]['kernel_size'] = st.slider(f'Kernel Size {i+1}', 1, 21, 5, step=2, key=f"kernel_{i}")
        elif filter_name == 'edge detection':
            st.session_state.filters[i]['threshold1'] = st.slider(f'Threshold1 {i+1}', 0, 300, 100, key=f"edge1_{i}")
            st.session_state.filters[i]['threshold2'] = st.slider(f'Threshold2 {i+1}', 0, 300, 200, key=f"edge2_{i}")
        elif filter_name == 'black and white':
            st.session_state.filters[i]['threshold'] = st.slider(f'Black & White Threshold {i+1}', 0, 255, 127, key=f"bw_threshold_{i}")
        elif filter_name == 'adjust rgb':
            st.session_state.filters[i]['r'] = st.slider(f'Red Channel {i+1}', 0.0, 3.0, 1.0, key=f"red_{i}")
            st.session_state.filters[i]['g'] = st.slider(f'Green Channel {i+1}', 0.0, 3.0, 1.0, key=f"green_{i}")
            st.session_state.filters[i]['b'] = st.slider(f'Blue Channel {i+1}', 0.0, 3.0, 1.0, key=f"blue_{i}")

    # Process the uploaded image
    if image_file is not None:
        image_pil = Image.open(image_file)
        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Apply selected filters
        processed_image_bgr = apply_filters(image_bgr, st.session_state.filters)
        processed_image_rgb = cv2.cvtColor(processed_image_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2, gap="small")
        
        with col1:
            st.markdown("<h3 class='center'>Original Image</h3>", unsafe_allow_html=True)
            st.image(image_pil, use_container_width=True)
            fig1 = plt.figure()
            st.pyplot(plot_histogram_image(image_np, fig1))

        with col2:
            st.markdown("<h3 class='center'>Processed Image</h3>", unsafe_allow_html=True)
            st.image(processed_image_rgb, use_container_width=True)
            fig2 = plt.figure()
            st.pyplot(plot_histogram_image(processed_image_rgb, fig2))

# ------------------ TAB 2: VIDEO INPUT --------------------#
with tabs[1]:
    st.subheader("Live Video Input")

    # Initialize video filters list
    if 'video_filters' not in st.session_state:
        st.session_state.video_filters = [{'name': 'default'}]

    # Add or Remove Filters for video
    if st.button("Add Filter", key="video_add"):
        st.session_state.video_filters.append({'name': 'default'})
        st.rerun(scope="app")  # Refresh the page after adding a video filter

    if len(st.session_state.video_filters) > 1 and st.button("Remove Filter", key="video_remove"):
        st.session_state.video_filters.pop()
        st.rerun(scope="app")  # Refresh the page after removing a video filter


    # Display filter options dynamically for video
    for i, filter_option in enumerate(st.session_state.video_filters):
        filter_name = st.selectbox(f"Select Video Filter {i+1}", 
                                   ['default', 'grayscale', 'black and white', 'negative',
                                    'brightness', 'contrast', 'gaussian blur', 'edge detection',
                                    'sepia', 'emboss', 'adjust rgb'],
                                   key=f"video_filter_select_{i}")
        st.session_state.video_filters[i]['name'] = filter_name

        # Add sliders based on filter type
        if filter_name == 'brightness':
            st.session_state.video_filters[i]['value'] = st.slider(f'Brightness Level {i+1}', -100, 100, 50, key=f"video_brightness_{i}")
        elif filter_name == 'contrast':
            st.session_state.video_filters[i]['value'] = st.slider(f'Contrast Level {i+1}', 0.1, 3.0, 1.5, key=f"video_contrast_{i}")
        elif filter_name == 'gaussian blur':
            st.session_state.video_filters[i]['kernel_size'] = st.slider(f'Kernel Size {i+1}', 1, 21, 5, step=2, key=f"video_kernel_{i}")
        elif filter_name == 'edge detection':
            st.session_state.video_filters[i]['threshold1'] = st.slider(f'Threshold1 {i+1}', 0, 300, 100, key=f"video_edge1_{i}")
            st.session_state.video_filters[i]['threshold2'] = st.slider(f'Threshold2 {i+1}', 0, 300, 200, key=f"video_edge2_{i}")
        elif filter_name == 'black and white':
            st.session_state.video_filters[i]['threshold'] = st.slider(f'Black & White Threshold {i+1}', 0, 255, 127, key=f"video_bw_threshold_{i}")
        elif filter_name == 'adjust rgb':
            st.session_state.video_filters[i]['r'] = st.slider(f'Red Channel {i+1}', 0.0, 3.0, 1.0, key=f"video_red_{i}")
            st.session_state.video_filters[i]['g'] = st.slider(f'Green Channel {i+1}', 0.0, 3.0, 1.0, key=f"video_green_{i}")
            st.session_state.video_filters[i]['b'] = st.slider(f'Blue Channel {i+1}', 0.0, 3.0, 1.0, key=f"video_blue_{i}")

    col1, col2 = st.columns(2, gap="small")

    with col1:
        st.markdown("<h3 class='center'>Original Video</h3>", unsafe_allow_html=True)
        input_image_placeholder = st.image([])
        input_hist_placeholder = st.pyplot(plt.figure())

    with col2:
        st.markdown("<h3 class='center'>Processed Video</h3>", unsafe_allow_html=True)
        output_image_placeholder = st.image([])
        output_hist_placeholder = st.pyplot(plt.figure())

    cap = cv2.VideoCapture(0)
    FRAME_RATE = 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from camera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image_placeholder.image(frame_rgb)

        # Plot histogram for the original video frame
        fig_input_hist = plt.figure()
        input_hist_placeholder.pyplot(plot_histogram_video(frame, fig_input_hist))

        # Apply selected filters for video
        processed_frame = apply_filters(frame, st.session_state.video_filters)
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        output_image_placeholder.image(processed_frame_rgb)

        # Plot histogram for the processed video frame
        fig_output_hist = plt.figure()
        output_hist_placeholder.pyplot(plot_histogram_video(processed_frame, fig_output_hist))

        time.sleep(1 / FRAME_RATE)

    cap.release()
