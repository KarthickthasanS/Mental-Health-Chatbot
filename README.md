# Project: Chatbot with Flask

## Description
This project is a simple chatbot built with Flask that can respond to user input based on pre-defined intents. It uses a machine learning model to classify user input and select an appropriate response from a set of predefined responses. The python version used for this project is Python 3.8

## Key Features
- Responds to user input with pre-defined answers.
- Uses a machine learning model for intent classification.
- Easy to extend with new intents and responses.

## Dependencies
- Flask
- NLTK
- Keras
- Numpy

## Setup Instructions
1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/your-username/your-repository.git
2. Install the required dependencies using pip.
    ```bash
    pip install -r requirements.txt
3. Download NLTK data for lemmatization.
   ```bash
    import nltk
    nltk.download('wordnet')
4. Run the Flask application.
   ```bash
    python app.py

Open your web browser and navigate to http://localhost:5000 to access the chatbot interface.
## Usage
Enter a message in the chat input box and press Enter.
The chatbot will process your message and provide a response based on the trained model.
