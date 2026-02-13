# Car Damage Severity Classification
This project demonstrates the application of machine learning to estimate vehicle damage severity from a single image. 
The model classifies the uploaded photo into three categories: minor, moderate, and severe. A simple Django web application 
is included to allow users to upload an image and view the predicted damage level.

## How to Run the Project
1. **Clone the repository**  
git clone https://github.com/filipkampic/car-damage-classification-ruap  
cd car-damage-classification-ruap

2. **Install dependencies**  
pip install -r requirements.txt

3. **Start the application**  
python manage.py runserver

The app will be available at: http://127.0.0.1:8000/

## Usage
1. Upload a vehicle image

2. The application processes the image and sends it to the model

3. The predicted damage category is displayed on the results page

## Model
The system uses a Multiclass Decision Forest model, selected for its stable performance and lower misclassification rate 
compared to other tested algorithms. The model was trained on a prepared dataset of vehicle images, including preprocessing and feature extraction steps.

## Authors
Filip Kampić

Leonardo Tomlinović


Project created as part of the *_Računarstvo usluga i analiza podataka_* course.
