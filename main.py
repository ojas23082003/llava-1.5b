import streamlit as st
import requests
from PIL import Image
import io
from groq import Groq
import base64
import os
from io import BytesIO
import time

from dotenv import load_dotenv
load_dotenv()


# os.environ['GROQ_API_KEY'] = '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'
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['GROQ_API_KEY'] = groq_api_key
client = Groq()
llava_model = 'llava-v1.5-7b-4096-preview'

# Title of the app
st.title("Image and Text Processing App")

# Input: Image upload
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

save_directory = './uploaded_images'

# Create directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Input: Text box for user input
user_text = st.text_area("Enter text input")

# Button to trigger the process
if st.button("Process"):
    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        image_name = uploaded_image.name

        save_path = os.path.join(save_directory, image_name)
        image.save(save_path)

        # Convert image to byte format for API
        # img_byte_arr = io.BytesIO()
        # image.save(img_byte_arr, format=image.format)
        # img_byte_arr = img_byte_arr.getvalue()

        # image_path = os.path.abspath('downloaded_image.jpg')
        image_path = save_path
        # print(image_path)

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        if user_text is None:
            user_text = "Describe this image"
        start_time = time.time()
        chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
            model=llava_model, 
            temperature=0.2,
            max_tokens=2048
        )

        end_time = time.time()  # End timing inference
        inference_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        st.write(chat_completion.choices[0].message.content)
        st.write(f"Inference Time: {inference_time_ms:.2f} milliseconds")
    else:
        st.write("Please upload an image and enter text.")
