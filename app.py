import streamlit as st
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.models import load_model

# Function for preprocessing image
def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

# Function for extracting features
def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

# Function for recommending fashion items
def recommend_fashion_items_cnn(input_image, all_features, all_image_names, model, top_n=4):
    # pre-process the input image and extract features
    preprocessed_img = preprocess_image(input_image)
    input_features = extract_features(model, preprocessed_img)

    # calculate similarities and find the top N similar images
    similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]
    similar_indices = np.argsort(similarities)[-top_n:]

    # filter out the input image index from similar_indices
    similar_indices = [idx for idx in similar_indices]# if idx != all_image_names.index(input_image_path)]
    # display the input image and recommended images
    recommended_images = [] # Add input image as the first recommendation

    for idx in similar_indices:
        recommedneded_image_path = os.path.join('model_creation/women_fashion_data', all_image_names[idx])
        recommended_images.append(recommedneded_image_path)
        
    return recommended_images



# directory path containing your images
image_directory = 'model_creation/women_fashion_data'

image_paths_list = [file for file in glob.glob(os.path.join(image_directory, '*.*')) if file.endswith(('.jpg', '.png', '.jpeg', 'webp'))]


base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)
    
all_features = []
all_image_names = []

for img_path in image_paths_list:
    preprocessed_img = preprocess_image(img_path)
    features = extract_features(model, preprocessed_img)
    all_features.append(features)
    all_image_names.append(os.path.basename(img_path))
    
def main():
    # Title and Description
    st.title('Women Fashion Recommendation System')
    st.subheader("Author : Rahul Bhoyar")
    st.write("Upload an image of the clothing item you want recommendations for:")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Get the path to the uploaded image
        input_image_path = "uploaded_image.jpg"  # Save the uploaded image temporarily
        image.save(input_image_path)

        # Display waiting message
        with st.spinner("Please wait while we process your request..."):
            # Display recommendation results
            st.write("Here are the recommended images based on the uploaded image:")
            
            # Call the recommend_fashion_items_cnn function
            recommended_images = recommend_fashion_items_cnn(input_image_path, all_features, all_image_names, model)

            # Display recommended images
            col1, col2 = st.columns(2)
            columns = [col1, col2]

            for i, recommended_image_path in enumerate(recommended_images, start=1):
                recommended_image = Image.open(recommended_image_path)
                with columns[i % 2]:
                    st.image(recommended_image, caption=f"Recommendation {i}", width=200, use_column_width='auto')
                    
    # Footer Section
    st.text('Powered by Streamlit')

if __name__ == '__main__':
    main()