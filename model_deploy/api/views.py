import pickle
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


tfidf = pickle.load(open('ml_model/vectorizer.pkl','rb'))
model = pickle.load(open('ml_model/model.pkl', 'rb'))

# Create your views here.
@api_view(['GET'])
def index_page(request):
    return_data = {
        "error" : "0",
        "message" : "Successful",
    }
    return Response(return_data)

def lower_case(text):
    return text.lower()

def text_preprocessing(text):
    # word tokenizing
    tokens = nltk.word_tokenize(text)

    # removing noise: numbers, stopwords, and punctuation
    stopwords_list = stopwords.words("english")
    tokens = [token for token in tokens if not token.isdigit() and \
                                token not in stopwords_list]
    
    n = nltk.WordNetLemmatizer()
    tokens = [n.lemmatize(token) for token in tokens]

    # join tokens and form string
    preprocessed_text = " ".join(tokens)

    return preprocessed_text



@api_view(["POST"])
def predict_spam(request):
    print('in server')
    data = request.get_json()
    email = data.get('message', '')
    print(email)
    email=[email]

    lower_email = lower_case(email)
    transformed_email= text_preprocessing(lower_email)
    vector_input = tfidf.transform([transformed_email])
    output = model.predict(vector_input)
    print(output)

    if output[1]=='spam':
        return Response({"val": True})
    else:
        return Response({"val": False})