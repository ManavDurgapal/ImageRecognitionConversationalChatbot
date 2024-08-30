# # import os
# # import json
# # import streamlit as st
# # from groq import Groq
# # from PIL import Image, UnidentifiedImageError, ExifTags
# # import requests
# # from io import BytesIO
# # from transformers import pipeline
# # from final_captioner import generate_final_caption

# # # streamlit page title
# # st.title("PicSamvaad : Image Conversational Chatbot")

# # # Load configuration
# # working_dir = os.path.dirname(os.path.abspath(__file__))
# # config_data = json.load(open(f"{working_dir}/config.json"))

# # GROQ_API_KEY = config_data["GROQ_API_KEY"]

# # # Save the API key to environment variable
# # os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# # client = Groq()

# # # Sidebar for image upload and URL input
# # with st.sidebar:
# #     st.header("Upload Image or Enter URL")

# #     uploaded_file = st.file_uploader(
# #         "Upload an image to chat...", type=["jpg", "jpeg", "png"]
# #     )
# #     url = st.text_input("Or enter a valid image URL...")

# # image = None
# # error_message = None


# # def correct_image_orientation(img):
# #     try:
# #         for orientation in ExifTags.TAGS.keys():
# #             if ExifTags.TAGS[orientation] == "Orientation":
# #                 break
# #         exif = img._getexif()
# #         if exif is not None:
# #             orientation = exif[orientation]
# #             if orientation == 3:
# #                 img = img.rotate(180, expand=True)
# #             elif orientation == 6:
# #                 img = img.rotate(270, expand=True)
# #             elif orientation == 8:
# #                 img = img.rotate(90, expand=True)
# #     except (AttributeError, KeyError, IndexError):
# #         pass
# #     return img


# # # Process uploaded image or URL
# # if uploaded_file is not None:
# #     image = Image.open(uploaded_file)
# #     image = correct_image_orientation(image)
# #     st.image(image, caption="Uploaded Image.", use_column_width=True)

# # if url:
# #     try:
# #         response = requests.get(url)
# #         response.raise_for_status()  # Check if the request was successful
# #         image = Image.open(BytesIO(response.content))
# #         image = correct_image_orientation(image)
# #         st.image(image, caption="Image from URL.", use_column_width=True)
# #     except (requests.exceptions.RequestException, UnidentifiedImageError) as e:
# #         image = None
# #         error_message = "Error: The provided URL is invalid or the image could not be loaded. Sometimes some image URLs don't work. We suggest you upload the downloaded image instead ;)"

# # caption = ""
# # if image is not None:
# #     caption += generate_final_caption(image)
# #     st.write("ChatBot : " + caption)

# # # Display error message if any
# # if error_message:
# #     st.error(error_message)

# # # Initialize chat history in Streamlit session state if not present already
# # if "chat_history" not in st.session_state:
# #     st.session_state.chat_history = []

# # # Display chat history
# # for message in st.session_state.chat_history:
# #     with st.chat_message(message["role"]):
# #         st.markdown(message["content"])

# # # Input field for user's message
# # user_prompt = st.chat_input("Ask the Chatbot about the image...")

# # if user_prompt:
# #     st.chat_message("user").markdown(user_prompt)
# #     st.session_state.chat_history.append({"role": "user", "content": user_prompt})

# #     # Send user's message to the LLM and get a response
# #     messages = [
# #         {
# #             "role": "system",
# #             "content": "You are a helpful, accurate image conversational assistant. You don't hallucinate, and your answers are very precise and have a positive approach. The caption of the image is: "
# #             + caption,
# #         },
# #         *st.session_state.chat_history,
# #     ]

# #     response = client.chat.completions.create(
# #         model="llama-3.1-8b-instant", messages=messages
# #     )

# #     assistant_response = response.choices[0].message.content
# #     st.session_state.chat_history.append(
# #         {"role": "assistant", "content": assistant_response}
# #     )

# #     # Display the LLM's response
# #     with st.chat_message("assistant"):
# #         st.markdown(assistant_response)


# import os
# import json
# import streamlit as st
# from groq import Groq
# from PIL import Image, UnidentifiedImageError, ExifTags
# import requests
# from io import BytesIO
# from transformers import pipeline
# from final_captioner import generate_final_caption

# # Streamlit page title
# st.title("PicSamvaad : Image Conversational Chatbot")

# # Load configuration
# working_dir = os.path.dirname(os.path.abspath(__file__))
# config_data = json.load(open(f"{working_dir}/config.json"))

# GROQ_API_KEY = config_data["GROQ_API_KEY"]

# # Save the API key to environment variable
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# client = Groq()

# # Sidebar for image upload and URL input
# with st.sidebar:
#     st.header("Upload Image or Enter URL")

#     uploaded_file = st.file_uploader(
#         "Upload an image to chat...", type=["jpg", "jpeg", "png"]
#     )
#     url = st.text_input("Or enter a valid image URL...")

# image = None
# error_message = None


# def correct_image_orientation(img):
#     try:
#         for orientation in ExifTags.TAGS.keys():
#             if ExifTags.TAGS[orientation] == "Orientation":
#                 break
#         exif = img._getexif()
#         if exif is not None:
#             orientation = exif[orientation]
#             if orientation == 3:
#                 img = img.rotate(180, expand=True)
#             elif orientation == 6:
#                 img = img.rotate(270, expand=True)
#             elif orientation == 8:
#                 img = img.rotate(90, expand=True)
#     except (AttributeError, KeyError, IndexError):
#         pass
#     return img


# # Check if a new image or URL has been provided and reset chat history
# if uploaded_file or url:
#     if "last_uploaded" not in st.session_state:
#         st.session_state.last_uploaded = None

#     if uploaded_file or (url and st.session_state.last_uploaded != url):
#         st.session_state.chat_history = []  # Clear chat history
#         st.session_state.last_uploaded = uploaded_file or url  # Update last uploaded

# # Process uploaded image or URL
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     image = correct_image_orientation(image)
#     st.image(image, caption="Uploaded Image.", use_column_width=True)

# if url:
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Check if the request was successful
#         image = Image.open(BytesIO(response.content))
#         image = correct_image_orientation(image)
#         st.image(image, caption="Image from URL.", use_column_width=True)
#     except (requests.exceptions.RequestException, UnidentifiedImageError) as e:
#         image = None
#         error_message = "Error: The provided URL is invalid or the image could not be loaded. Sometimes some image URLs don't work. We suggest you upload the downloaded image instead ;)"

# caption = ""
# if image is not None:
#     caption += generate_final_caption(image)
#     st.write("ChatBot : " + caption)

# # Display error message if any
# if error_message:
#     st.error(error_message)

# # Initialize chat history in Streamlit session state if not present already
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Display chat history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Input field for user's message
# user_prompt = st.chat_input("Ask the Chatbot about the image...")

# if user_prompt:
#     st.chat_message("user").markdown(user_prompt)
#     st.session_state.chat_history.append({"role": "user", "content": user_prompt})

#     # Send user's message to the LLM and get a response
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful, accurate image conversational assistant. You don't hallucinate, and your answers are very precise and have a positive approach. The caption of the image is: "
#             + caption,
#         },
#         *st.session_state.chat_history,
#     ]

#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant", messages=messages
#     )

#     assistant_response = response.choices[0].message.content
#     st.session_state.chat_history.append(
#         {"role": "assistant", "content": assistant_response}
#     )

#     # Display the LLM's response
#     with st.chat_message("assistant"):
#         st.markdown(assistant_response)
import os
import json
import streamlit as st
from groq import Groq
from PIL import Image, UnidentifiedImageError, ExifTags
import requests
from io import BytesIO
from transformers import pipeline
from final_captioner import generate_final_caption
import hashlib

# Streamlit page title
st.title("PicSamvaad : Image Conversational Chatbot")

# Load configuration
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))

GROQ_API_KEY = config_data["GROQ_API_KEY"]

# Save the API key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = Groq()

# Sidebar for image upload and URL input
with st.sidebar:
    st.header("Upload Image or Enter URL")

    uploaded_file = st.file_uploader(
        "Upload an image to chat...", type=["jpg", "jpeg", "png"]
    )
    url = st.text_input("Or enter a valid image URL...")

image = None
error_message = None


def correct_image_orientation(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = img._getexif()
        if exif is not None:
            orientation = exif[orientation]
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return img


def get_image_hash(image):
    # Generate a unique hash for the image
    img_bytes = image.tobytes()
    return hashlib.md5(img_bytes).hexdigest()


# Check if a new image or URL has been provided and reset chat history
if "last_uploaded_hash" not in st.session_state:
    st.session_state.last_uploaded_hash = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_hash = get_image_hash(image)

    if st.session_state.last_uploaded_hash != image_hash:
        st.session_state.chat_history = []  # Clear chat history
        st.session_state.last_uploaded_hash = image_hash  # Update last uploaded hash

    image = correct_image_orientation(image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

elif url:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        image = Image.open(BytesIO(response.content))
        image_hash = get_image_hash(image)

        if st.session_state.last_uploaded_hash != image_hash:
            st.session_state.chat_history = []  # Clear chat history
            st.session_state.last_uploaded_hash = (
                image_hash  # Update last uploaded hash
            )

        image = correct_image_orientation(image)
        st.image(image, caption="Image from URL.", use_column_width=True)
    except (requests.exceptions.RequestException, UnidentifiedImageError) as e:
        image = None
        error_message = "Error: The provided URL is invalid or the image could not be loaded. Sometimes some image URLs don't work. We suggest you upload the downloaded image instead ;)"

caption = ""
if image is not None:
    caption += generate_final_caption(image)
    st.write("ChatBot : " + caption)

# Display error message if any
if error_message:
    st.error(error_message)

# Initialize chat history in Streamlit session state if not present already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user's message
user_prompt = st.chat_input("Ask the Chatbot about the image...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Send user's message to the LLM and get a response
    messages = [
        {
            "role": "system",
            "content": "You are a helpful, accurate image conversational assistant. You don't hallucinate, and your answers are very precise and have a positive approach. The caption of the image is: "
            + caption,
        },
        *st.session_state.chat_history,
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", messages=messages
    )

    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_response}
    )

    # Display the LLM's response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
