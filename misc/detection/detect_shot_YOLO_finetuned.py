from roboflow import Roboflow
# need to use API, not going up on Github
API_KEY = ""
PROJECT_NAME = "throws-vision"  
VERSION = 1  

rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project(PROJECT_NAME)
model = project.version(VERSION).model

# Infer on a local image
print(model.predict("./images/basic.jpg", confidence=40, overlap=30).json())

# Visualize your prediction
model.predict("./images/basic.jpg", confidence=40, overlap=30).save("prediction.jpg")

# Infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
