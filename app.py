
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os

st.set_page_config(page_title="Klasifikasi Sampah", page_icon="♻️")

st.title("♻️ Klasifikasi Sampah Otomatis")
st.write("Upload gambar sampah untuk mengetahui jenisnya!")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('garbage_classifier_model.h5', compile=False)
    with open('class_indices.pkl', 'rb') as f:
        class_indices = pickle.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_model()

uploaded_file = st.file_uploader("Pilih gambar...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)
    
    if st.button('Klasifikasikan!'):
        # Preprocess
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        # Result
        st.success(f"**Kategori:** {idx_to_class[predicted_class]}")
        st.metric("Confidence", f"{confidence:.2f}%")
        
        # Top 5
        st.write("### Top 5 Prediksi:")
        top_5 = np.argsort(predictions[0])[-5:][::-1]
        for i in top_5:
            st.write(f"- {idx_to_class[i]}: {predictions[0][i]*100:.2f}%")
