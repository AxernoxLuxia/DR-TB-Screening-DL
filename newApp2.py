import streamlit as st
import numpy as np
import time as t
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

with tf.device('/GPU:0'):
    efficientnetB0_model = tf.keras.models.load_model('efficientnetb0.keras', compile=False)
    unet_model = tf.keras.models.load_model('Lung_Segmentation/lung_segmentation.keras', compile=False)

efficientnetB0_model.compile()
unet_model.compile()

def clahe(image):
    image = image.convert("RGB")
    image = keras.utils.img_to_array(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4)
    clahe_img = clahe.apply(image.astype('uint8'))
    return clahe_img

def gabor(image):
    image = keras.utils.img_to_array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = np.squeeze(image)
    ksize = 30  # Kernel size
    sigma = 1.414  # Standard deviation of the Gaussian function
    lambd = 4.0  # Wavelength of the sinusoidal factor
    gamma = 1  # Spatial aspect ratio
    psi = 0  # Phase offset
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    combined_image = np.zeros_like(image, dtype=np.float32)
    for theta in orientations:
        gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        filtered_image = cv2.filter2D(image, cv2.CV_32F, gabor_kernel)
        combined_image += filtered_image
    combined_image = cv2.normalize(combined_image, None, 0, 255, cv2.NORM_MINMAX)
    combined_image = np.uint8(combined_image)
    return combined_image

def canny(image):
    image = keras.utils.img_to_array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.astype(np.uint8)
    edges = cv2.Canny(image,100,200)
    return edges

def lung_cropping(image):
    xray_resized = cv2.resize(image, (256, 256))
    xray_input = np.expand_dims(xray_resized / 255.0, axis=0)

    pred_mask = unet_model.predict(xray_input)[0, :, :, 0] 
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    _, binary_mask = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)

    y_indices, x_indices = np.where(binary_mask > 0)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    cropped_xray = xray_resized[y_min:y_max, x_min:x_max]

    return cropped_xray

def preprocess_image(image, target_size=(256, 256)):
    print(image.shape)
    image = Image.fromarray(image, mode='L')
    image = image.convert("RGB")
    image = image.resize(target_size)
    return np.expand_dims(image, axis=0) 

def get_prediction(image):
    pred = efficientnetB0_model.predict(image)  # (1, 256, 256, 3)
    print('***********')
    print(pred)
    class_names = ['Resistive', 'Sensitive']
    confidence = np.max(pred) * 100
    label = class_names[np.argmax(np.round(pred,2))]
    return label, confidence

#Streamlit UI
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(to bottom right, white, #94BAD9) 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center; color:#456B87;'>Diagnosis of Drug Resistivity of Tuberculosis</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align:justify; color:#000000; font-size:20px'>This WebApp leverages the power of deep learning to assist in the early detection of Tuberculosis (TB) from chest X-ray images. Built with a trained EfficientNetB0 model, the app classifies uploaded chest X-rays into Drug-Sensitive TB and Drug-Resistant TB.<br>\
        It showcases the image processing which we do on the uploaded image. Image processing includes applying the filters like CLAHE, Gabor, Canny-Edge and on top of CLAHE, how we are doing the Lung Cropping.\
        Through this, the user can compare how the filters and the Lung Cropping affect the image and how the filters lilke Gabor and Canny-edge are not viable for this dataset.<br>\
        The final result is the prediction of the model (EfficientNetB0) which is either Drug-Sensitive TB or Drug-Resistant TB.</p>", 
    unsafe_allow_html=True)


img_upload = st.file_uploader("Insert a JPG image", type='jpg')  
if img_upload is not None:
    image = Image.open(img_upload)
    img_array = np.array(image)

    st.markdown(
        """
        <style>
        .centered-image {
            display: flex;
            justify-content: center;   
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="centered-image">', unsafe_allow_html=True)
    st.image(img_array, caption="Uploaded Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("#### Model selected: EfficientNetB0",)

    with st.spinner("Loading"):
        t.sleep(2)

    st.markdown("""
        <style>
        .stButton>button {
            display: block;
            margin-left: auto;
            margin-right: auto;
            background-color: #CCE6FF; 
            color: black; 
            padding: 10px 24px;
            font-size: 16px;
            border-radius: 8px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #E5F2FF; 
        }
        </style>
        """, unsafe_allow_html=True)
 
    if st.button("Predict"):

        st.text("")
        st.text("")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        img_clahe = clahe(image)
        img_gabor = gabor(img_clahe)
        img_canny = canny(img_gabor)
        lungs_cropped = lung_cropping(img_clahe)
        with filter_col1:
            st.image(img_clahe, caption="CLAHE Filter", use_container_width=True)
        
        with filter_col2:
            st.image(img_gabor, caption="GABOR Filter", use_container_width=True)
        
        with filter_col3:
            st.image(img_canny, caption="CANNY-EDGE Filter", use_container_width=True)

        st.text("")
        
        st.markdown('<div class="centered-image">', unsafe_allow_html=True)
        st.image(lungs_cropped, caption="Lungs Cropped", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        preprocessed_image = preprocess_image(lungs_cropped)
        label, confidence = get_prediction(preprocessed_image)
        st.text("")
        st.text("")
        with st.spinner("Loading"):
            t.sleep(2)
        st.markdown(f"<p style='text-align:center; color:#456B87; font-size:20px'>Prediction: {label}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; color:#456B87; font-size:20px'>Confidence score: {confidence:.2f}</p>", unsafe_allow_html=True)
