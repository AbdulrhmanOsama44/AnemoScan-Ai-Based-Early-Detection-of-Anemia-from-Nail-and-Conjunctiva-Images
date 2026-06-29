import os
import cv2
import sys
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utilities
from utils.pdf_generator import create_pdf
from utils.roi_detection import get_detector
from utils.preprocess import preprocess_image
from utils.nutrition import get_nutrition_advice
from utils.gradcam import get_gradcam_heatmap, overlay_heatmap_masked

# Page config
st.set_page_config(page_title = 'AnemoScan', page_icon = '🩸', layout = 'wide')

# Sidebar (disclaimers, info)
with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
    if os.path.exists(logo_path):
        st.image(logo_path, width = 225)
    st.markdown('## Disclaimer ❗️')
    st.info(
        '- This system is not a medical diagnostic tool and is intended for educational purposes only.\n'
        '- Always consult a qualified healthcare professional.'
    )
    st.markdown('---')
    st.markdown('How to use:')
    st.markdown('1. Upload a conjunctiva or nail image\n'
                '2. Click Predict\n'
                '3. Explore the tabs for results, heatmap, nutrition, and report download')
    st.markdown('---')
    st.caption('AnemoScan v1.0 - TM471 Final Year Project')

# Main area title
st.title('🩺 AnemoScan')
st.markdown('### AI‑based Anemia Detection System')

# Load models (cached)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'Models')

@st.cache_resource
def load_models():
    conjunctiva_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best_initial_model_conjunctiva.keras'), compile = False)
    nails_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best_initial_model_nails.keras'), compile = False)
    type_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'type_classifier_model.keras'), compile = False)
    return conjunctiva_model, nails_model, type_model

# Load YOLO detector (cached)
@st.cache_resource
def load_yolo():
    return get_detector(model_path = os.path.join(MODEL_DIR, 'best.pt'), conf_threshold = 0.5)

# Show loading spinner while models load for the first time
with st.spinner('🔄 Loading AI models... This may take a few seconds.'):
    conj_model, nail_model, type_model = load_models()
    yolo_detector = load_yolo()
st.success('✅ Models loaded successfully!')

# Layout: two columns for upload and preview
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader('📤 Upload Image')
    uploaded_file = st.file_uploader('Choose an image (JPG, JPEG, PNG)', type = ['jpg', 'jpeg', 'png'])

with col2:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption = 'Uploaded Image', width = 400)
        # Preprocess the whole image for type classification only (remains original)
        img_np = np.array(image)
        normalized, img_resized = preprocess_image(img_np)
        img_array = np.expand_dims(normalized, axis = 0)
        st.success('✅ Image ready')
    else:
        st.markdown(
            """
            <div style="height: 97px; display: flex; align-items: center;">
            </div>
            """,
            unsafe_allow_html = True
        )
        st.info('👈 Upload an image to begin')

# Run YOLO detection after upload and store the cropped ROI in session state
if uploaded_file is not None:
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    with st.spinner('🔍 Detecting region of interest...'):
        detections = yolo_detector.detect(img_bgr)
    
    if detections:
        best = max(detections, key = lambda x: x['confidence'])
        cropped_bgr = best['cropped_image']
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        st.session_state['cropped_roi'] = cropped_bgr   # store BGR for later use
        st.session_state['yolo_class'] = best['class_name']
        st.session_state['yolo_conf'] = best['confidence']

        # Get original file name without extension
        original_name = os.path.splitext(uploaded_file.name)[0]
        roi_filename = f"{original_name}_{best['class_name']}.jpg"
        
        # Display downloadable ROI (collapsible)
        with st.expander('🔎 Detected ROI (YOLO) – Click to view', expanded = False):
            col_roi1, col_roi2 = st.columns([1, 2])
            with col_roi1:
                st.image(cropped_rgb, caption = f"{best['class_name'].capitalize()} (conf: {best['confidence']:.1%})", width = 250)
            with col_roi2:
                _, buffer = cv2.imencode('.jpg', cropped_bgr)
                roi_bytes = buffer.tobytes()
                st.download_button(
                    label = '📥 Download Cropped ROI',
                    data = roi_bytes,
                    file_name = roi_filename,
                    mime = 'image/jpeg'
                )
                st.caption('This region will be used for anemia prediction and Grad‑CAM.')
    else:
        st.warning('⚠️ YOLO could not detect any conjunctiva or nail. Please try a clearer image.')
        st.session_state['cropped_roi'] = None

# Prediction button (only if image uploaded)
if uploaded_file is not None:
    if st.button('🔍 Predict', type = 'primary', use_container_width = True):

        with st.spinner('🧠 Analyzing image... This may take a few seconds.'):

            # Type Classification
            type_pred = type_model.predict(img_array)[0][0]
            type_confidence = max(type_pred, 1 - type_pred)
            THRESHOLD = 0.75

            if type_confidence < THRESHOLD:
                st.error('❌ Unable to determine image type confidently. Please upload a clear conjunctiva or nail image.')
                st.stop()

            # Select the anemia model based on type classifier
            if type_pred > 0.5:
                image_type = 'Nails'
                anemia_model = nail_model
            else:
                image_type = 'Conjunctiva'
                anemia_model = conj_model

            # Use cropped ROI for Anemia prediction & GRAD-CAM
            if st.session_state.get('cropped_roi') is not None:
                # Preprocess the cropped ROI (BGR -> RGB -> resize 224 -> normalize)
                roi_bgr = st.session_state['cropped_roi']
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                roi_normalized, roi_resized = preprocess_image(roi_rgb)   # (224,224,3) RGB
                roi_array = np.expand_dims(roi_normalized, axis = 0)
            else:
                # Fallback to whole image if YOLO failed (should not happen if we warned)
                roi_array = img_array
                roi_resized = img_resized

            # Anemia prediction on the ROI
            prediction = anemia_model.predict(roi_array)[0][0]
            if prediction > 0.5:
                label = 'Non-Anemic'
                confidence = prediction
            else:
                label = 'Anemic'
                confidence = 1 - prediction

            # GRAD-CAM on the ROI (resized image)
            heatmap = get_gradcam_heatmap(anemia_model, roi_array, 'Conv_1')
            overlay = overlay_heatmap_masked(np.array(roi_resized), heatmap)   # BGR overlay

            # Nutrition Advice
            advice = get_nutrition_advice(label)

            # Prepare PDF
            original_path = 'temp_original.png' # cropped ROI
            heatmap_path = 'temp_heatmap.png'
            full_original_path = 'temp_full_original.png'   # full uploaded image

            cv2.imwrite(original_path, cv2.cvtColor(roi_resized, cv2.COLOR_RGB2BGR))
            cv2.imwrite(heatmap_path, overlay)

            # Save the full original uploaded image (PIL image -> BGR)
            full_img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(full_original_path, full_img_bgr)

            # Get uploaded image name (without extension)
            image_name = os.path.splitext(uploaded_file.name)[0]
            heatmap_name = f'{image_name}_Heatmap'
            report_name = f'{image_name}_Report'

            # Generate PDF
            pdf_path = create_pdf(
                prediction = label,
                confidence = confidence,
                image_type = image_type,
                original_path = original_path,
                heatmap_path = heatmap_path,
                nutrition_text = advice,
                report_name = report_name,
                full_original_path = full_original_path
            )

            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()

            # Cleanup temps
            os.remove(original_path)
            os.remove(heatmap_path)
            os.remove(full_original_path)

            # Store results in session state
            st.session_state['results'] = {
                'image_type': image_type,
                'type_confidence': type_confidence,
                'label': label,
                'confidence': confidence,
                'overlay': overlay,          # BGR heatmap overlay (224x224)
                'advice': advice,
                'pdf_bytes': pdf_bytes,
                'report_name': report_name,
                'heatmap_name': heatmap_name,
            }
        st.session_state['prediction_done'] = True
        st.rerun()

# Display results using tabs (only if prediction done)
if st.session_state.get('prediction_done', False):
    res = st.session_state['results']

    st.markdown('---')
    st.subheader('📊 Analysis Results')

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(['📈 Results', '🔥 Grad‑CAM', '🥗 Nutrition', '📄 Download'])

    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric('Detected Image Type', res['image_type'])
        col2.metric('Type Confidence', f"{res['type_confidence']:.1%}")
        col3.metric('Prediction Confidence', f"{res['confidence']:.1%}")

        if res['label'] == 'Anemic':
            st.error(f"⚠️ {res['label']} detected")
        else:
            st.success(f"✅ {res['label']}")

        st.caption('This result is AI‑based and should not replace professional medical advice.')

        st.markdown('### 📷 Image Type Confidence Level')
        type_conf_value = float(res['type_confidence'])
        type_conf_value = max(0.0, min(1.0, type_conf_value))
        st.progress(type_conf_value)
        if type_conf_value >= 0.85:
            st.success('🟢 High Confidence – Image type clearly identified')
        elif type_conf_value >= 0.65:
            st.warning('🟡 Moderate Confidence – Check if image is a clear conjunctiva / nail')
        else:
            st.error('🔴 Low Confidence – Image may be unclear or wrong type')

        st.markdown('### 🧠 Model Confidence Level')
        confidence_value = float(res['confidence'])
        confidence_value = max(0.0, min(1.0, confidence_value))
        st.progress(confidence_value)
        if confidence_value >= 0.85:
            st.success('🟢 High Confidence – Model is very certain')
        elif confidence_value >= 0.65:
            st.warning('🟡 Moderate Confidence – Interpret with caution')
        else:
            st.error('🔴 Low Confidence – Result may be unreliable')

    with tab2:
        st.image(res['overlay'], caption = 'Grad‑CAM Heatmap (on cropped ROI)', width = 400, channels = 'BGR')
        if res['label'] == 'Anemic':
            st.info('The heatmap highlights regions that contributed to the detection of anemia.')
        else:
            st.info('The heatmap highlights regions the model used to confirm the absence of anemia‑related features.')
        
        _, buffer = cv2.imencode('.png', res['overlay'])
        img_bytes = buffer.tobytes()
        st.download_button(
            label = '📥 Download Heatmap',
            data = img_bytes,
            file_name = f"{res['heatmap_name']}_{res['label']}.png",
            mime = 'image/png'
        )

    with tab3:
        st.markdown(res['advice'])

    with tab4:
        st.download_button(
            label = '📄 Download Full PDF Report (Professional)',
            data = res['pdf_bytes'],
            file_name = f"{res['report_name']}.pdf",
            mime = 'application/pdf'
        )
