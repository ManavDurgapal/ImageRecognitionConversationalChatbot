# # -*- coding: utf-8 -*-
# from google.colab import drive
# drive.mount('/content/drive')

# from tensorflow.keras.models import load_model

# !pip install transformers
# !pip install PIL
# !pip install requests
# !pip install transformers torch


from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from transformers import pipeline
import gdown
import os

git_pipe = pipeline("image-to-text", model="microsoft/git-large-textcaps")


# file_id = "1PXixJsrUaVcHEEC-jDlv4tHT2qrCrf5c"  # Replace with your file ID
# url = f"https://drive.google.com/uc?id={file_id}"

# # Output path to save the model
# output = "LandmarkClassifierV5.h5"  # Replace with your model file name

# # Download the file if it doesn't exist
# if not os.path.exists(output):
#     gdown.download(url, output, quiet=False)


# # Load your model
# def load_model(output):
#     # Replace with code to load your actual model
#     loaded_model = tf.keras.models.load_model(output)
#     return loaded_model


# model = load_model()

# # Print model summary to verify
# model.summary()

flower_output = "Flower_classifier.h5"
flower_model_id = "1AlBunIPDg4HYYCqhcHtOiXxnPFhmsoSn"
flower_url = f"https://drive.google.com/uc?id={flower_model_id}"
if not os.path.exists(flower_output):
    gdown.download(flower_url, flower_output, quiet=False)
flower_model = load_model(flower_output)
flower_model.summary()


bird_output = "Bird_classifier.h5"
bird_model_id = "1a6vqFERbrr_Cw-NyBqVHG7fsjU2-xKJ4"
bird_url = f"https://drive.google.com/uc?id={bird_model_id}"
if not os.path.exists(bird_output):
    gdown.download(bird_url, bird_output, quiet=False)
bird_model = load_model(bird_output)
bird_model.summary()


dog_output = "DogClassifier.h5"
dog_model_id = "1UFn1NGVtP5rhvcWnAANQ_4E9YRJvDEad"
dog_url = f"https://drive.google.com/uc?id={dog_model_id}"
if not os.path.exists(dog_output):
    gdown.download(dog_url, dog_output, quiet=False)
dog_model = load_model(dog_output)
dog_model.summary()


landmark_output = "LandmarkClassifierV5.h5"
landmark_model_id = "1PXixJsrUaVcHEEC-jDlv4tHT2qrCrf5c"  # Replace with your file ID
landmark_url = f"https://drive.google.com/uc?id={landmark_model_id}"
if not os.path.exists(landmark_output):
    gdown.download(landmark_url, landmark_output, quiet=False)
landmark_model = load_model(landmark_output)
landmark_model.summary()


# landmark_classifier_model_path = (
#     "Models\Bird_classifier.h5"
# )
# landmark_classifier_model = load_model(landmark_classifier_model_path)

# generate_caption_url('https://i.ytimg.com/vi/-ylolmt2e6o/maxresdefault.jpg')

# generate_caption_url('https://t3.gstatic.com/licensed-image?q=tbn:ANd9GcR8UMpkAgxoNVzMeGKv-LoQ4yhgaDoWmMZsemrowtqyy8B5m34IT_tNrcRmphzWLUky')

# generate_caption_url('https://www.akc.org/wp-content/uploads/2017/11/Labrador-Retrievers-three-colors.jpg')

dog_list = [
    "Bulldog",
    "Chihuahua (dog breed)",
    "Dobermann",
    "German Shepherd",
    "Golden Retriever",
    "Husky",
    "Labrador Retriever",
    "Pomeranian dog",
    "Pug",
    "Rottweiler",
    "Street dog",
]
flower_list = [
    "Jasmine",
    "Lavender",
    "Lily",
    "Lotus",
    "Orchid",
    "Rose",
    "Sunflower",
    "Tulip",
    "daisy",
    "dandelion",
]
bird_list = [
    "Crow",
    "Eagle",
    "Flamingo",
    "Hummingbird",
    "Parrot",
    "Peacock",
    "Pigeon",
    "Sparrow",
    "Swan",
]
landmark_list = [
    "The Agra Fort",
    "Ajanta Caves",
    "Alai Darwaza",
    "Amarnath Temple",
    "The Amber Fort",
    "Basilica of Bom Jesus",
    "Brihadisvara Temple",
    "Charar-e-Sharief shrine",
    "Charminar",
    "Chhatrapati Shivaji Terminus",
    "Chota Imambara",
    "Dal Lake",
    "The Elephanta Caves",
    "Ellora Caves",
    "Fatehpur Sikri",
    "Gateway of India",
    "Ghats in Varanasi",
    "Gol Gumbaz",
    "Golden Temple",
    "Group of Monuments at Mahabalipuram",
    "Hampi",
    "Hawa Mahal",
    "Humayun's Tomb",
    "The India gate",
    "Iron Pillar",
    "Jagannath Temple, Puri",
    "Jageshwar",
    "Jama Masjid",
    "Jamali Kamali Tomb",
    "Jantar Mantar, Jaipur",
    "Jantar Mantar, New Delhi",
    "Kedarnath Temple",
    "Khajuraho Temple",
    "Konark Sun Temple",
    "Mahabodhi Temple",
    "Meenakshi Temple",
    "Nalanda mahavihara",
    "Parliament House, New Delhi",
    "Qutb Minar",
    "Qutb Minar Complex",
    "Ram Mandir",
    "Rani ki Vav",
    "Rashtrapati Bhavan",
    "The Red Fort",
    "Sanchi",
    "Supreme Court of India",
    "Swaminarayan Akshardham (Delhi)",
    "Taj Hotels",
    "The Lotus Temple",
    "The Mysore Palace",
    "The Statue of Unity",
    "The Taj Mahal",
    "Vaishno Devi Temple",
    "Venkateswara Temple, Tirumala",
    "Victoria Memorial, Kolkata",
    "Vivekananda Rock Memorial",
]


def identify_dog(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Get predictions
    predictions = dog_model.predict(img_array)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])

    # Get the probability of the predicted class
    predicted_probability = predictions[0][predicted_class_index]

    # Map the predicted class index to the class label
    predicted_class_label = dog_list[predicted_class_index]

    return predicted_class_label


# def generate_dog_url(img_url):
#     # Fetch the image from the URL
#     response = requests.get(img_url)
#     img = Image.open(BytesIO(response.content))

#     # Resize and preprocess the image
#     img = img.resize((224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     # Get predictions
#     predictions = dog_classifier_model.predict(img_array)
#     predicted_class_index = np.argmax(predictions[0])

#     # Map the predicted class index to the class label
#     return dog_list[predicted_class_index]


def identify_flower(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Get predictions
    predictions = flower_model.predict(img_array)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])

    # Get the probability of the predicted class
    predicted_probability = predictions[0][predicted_class_index]

    # Map the predicted class index to the class label
    predicted_class_label = flower_list[predicted_class_index]

    return predicted_class_label


# def generate_flower_url(img_url):
#     # Fetch the image from the URL
#     response = requests.get(img_url)
#     img = Image.open(BytesIO(response.content))

#     # Resize and preprocess the image
#     img = img.resize((224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     # Get predictions
#     predictions = flower_classifier_model.predict(img_array)
#     predicted_class_index = np.argmax(predictions[0])

#     # Map the predicted class index to the class label
#     return flower_list[predicted_class_index]


def identify_bird(img):
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Get predictions
    predictions = bird_model.predict(img_array)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])

    # Get the probability of the predicted class
    predicted_probability = predictions[0][predicted_class_index]

    # Map the predicted class index to the class label
    predicted_class_label = bird_list[predicted_class_index]

    return predicted_class_label


def identify_landmark(img):
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Get predictions
    predictions = landmark_model.predict(img_array)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])

    # Get the probability of the predicted class
    predicted_probability = predictions[0][predicted_class_index]

    # Map the predicted class index to the class label
    predicted_class_label = landmark_list[predicted_class_index]

    return predicted_class_label


# def generate_bird_url(img_url):
#     # Fetch the image from the URL
#     response = requests.get(img_url)
#     img = Image.open(BytesIO(response.content))

#     # Resize and preprocess the image
#     img = img.resize((224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     # Get predictions
#     predictions = bird_classifier_model.predict(img_array)
#     predicted_class_index = np.argmax(predictions[0])

#     # Map the predicted class index to the class label
#     return bird_list[predicted_class_index]


# def generate_landmark_path(img_path):
#     # Load and preprocess the image
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     # Get predictions
#     predictions = landmark_classifier_model.predict(img_array)
#     predicted_class_index = np.argmax(predictions[0])

#     # Map the predicted class index to the class label
#     return landmark_list[predicted_class_index]


# def generate_landmark_url(img_url):
#     # Fetch the image from the URL
#     response = requests.get(img_url)
#     img = Image.open(BytesIO(response.content))

#     # Resize and preprocess the image
#     img = img.resize((224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     # Get predictions
#     predictions = landmark_classifier_model.predict(img_array)
#     predicted_class_index = np.argmax(predictions[0])

#     # Map the predicted class index to the class label
#     return landmark_list[predicted_class_index]


# generate_landmark_url(
#     "https://miro.medium.com/v2/resize:fit:1400/1*VhDpUbuZQC4tLoP9qQ6W5A.jpeg"
# )

# generate_bird_url(
#     "https://images.pexels.com/photos/1406506/pexels-photo-1406506.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500.jpg"
# )


# def generate_caption_upload(upload):
#     # Read the image
#     img = Image.open(upload)
#     caption_dict = git_pipe(img)
#     caption = caption_dict[0]["generated_text"]
#     keyword_to_function = {
#         "dog": generate_dog_path,
#         "bird": generate_bird_path,
#         "flower": generate_flower_path,
#     }
#     for keyword, function in keyword_to_function.items():
#         if keyword in caption.lower():
#             specific_name = function(upload)
#             caption = caption.replace(keyword, specific_name)
#     phrase_to_cut = "with the word"
#     index = caption.find(phrase_to_cut)
#     result = caption[:index].strip() if index != -1 else caption
#     plt.imshow(img)
#     plt.axis("off")
#     plt.title(result)
#     plt.show()
#     return result


# def generate_caption_url(img_url):
#     response = requests.get(img_url)
#     image = Image.open(BytesIO(response.content)).convert("RGB")
#     # Extracting the caption text from the dictionary
#     caption_dict = git_pipe(image)
#     caption = caption_dict[0]["generated_text"]
#     keyword_to_function = {
#         "dog": generate_dog_url,
#         "bird": generate_bird_path,
#         "flower": generate_flower_path,
#     }
#     for keyword, function in keyword_to_function.items():
#         if keyword in caption.lower():
#             # Run the specific model function
#             specific_name = function(img_url)
#             # Replace the keyword with the identified specific name in the caption
#             caption = caption.replace(keyword, specific_name)
#     phrase_to_cut = "with the word"
#     index = caption.find(phrase_to_cut)
#     result = caption[:index].strip() if index != -1 else caption
#     plt.imshow(image)
#     plt.title(result)
#     plt.axis("off")
#     plt.show()
#     return result


def generate_final_caption(image):
    caption_dict = git_pipe(image)
    caption = caption_dict[0]["generated_text"]
    image = image.resize((256, 256))
    caption = caption_dict[0]["generated_text"]
    phrases_to_cut = ["with the word", "that says"]
    for phrase in phrases_to_cut:
        index = caption.find(phrase)
        if index != -1:
            caption = caption[:index].strip()
    # Check if the word "building" is in the caption
    if "building" in caption.lower():
        caption += "\nThe landmark is : " + identify_landmark(image)
    elif "flower" in caption.lower():
        caption += "\nThe Flower is : " + identify_flower(image)
    elif "dog" in caption.lower() or "puppy" in caption.lower():
        caption += "\nThe Dog is : " + identify_dog(image)
    elif "birds" in caption.lower() or "bird" in caption.lower():
        caption +="\nThe Bird is : "+identify_bird(image)
    return caption
