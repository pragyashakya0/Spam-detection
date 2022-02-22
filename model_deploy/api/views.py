import pickle
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
import numpy as np
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

tfidf = pickle.load(open('ml_model/vectorizer.pkl', 'rb'))
model = pickle.load(open("ml_model/model.pkl", 'rb'))

# Create your views here.


@api_view(['GET'])
def index_page(request):
    return_data = {
        "error": "0",
        "message": "Successful",
    }
    return Response(return_data)


def lower_case(text):
    return text.lower()


def remove_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def remove_username(text):
    return re.sub('@[^\s]+', '', text)


def remove_urls(text):
    return re.sub(r"((http\S+)|(www\.))", '', text)


def remove_special_characters(text):
    pattern = r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text


def remove_single_char(text):
    return re.sub(r'\b[a-zA-Z]\b', '', text)


def remove_singlespace(text):
    return re.sub(r'\s+', ' ', text)


def remove_whitespace(text):
    return re.sub(r'^\s+|\s+?$', '', text)


def remove_multiple(text):
    return re.sub("(.)\\1{2,}", "\\1", text)


def text_preprocessing(text):
    # word tokenizing
    tokens = nltk.word_tokenize(text)

    # removing noise: numbers, stopwords, and punctuation
    stopwords_list = stopwords.words("english")
    tokens = [token for token in tokens if not token.isdigit() and
              not token in string.punctuation and
              token not in stopwords_list]
    n = nltk.WordNetLemmatizer()
    tokens = [n.lemmatize(token) for token in tokens]

    # join tokens and form string
    preprocessed_text = " ".join(tokens)

    return preprocessed_text


@api_view(["POST"])
def predict_spam(request):
    # print('in server')
    email = request.data['email']
    # return Response({"data":"Hi"})

    # data = request.get_json()
    # email = data.get('message', '')
    # print(email)
    # email = [email]
    lower_email = lower_case(email)
    square_email = remove_square_brackets(lower_email)
    user_email = remove_username(square_email)
    urls_email = remove_urls(user_email)
    special_characters_email = remove_special_characters(urls_email)
    single_char_email = remove_single_char(special_characters_email)
    multiple_email = remove_multiple(single_char_email)
    singlespace_email = remove_singlespace(multiple_email)
    whitespace_email = remove_whitespace(singlespace_email)
    transformed_email = text_preprocessing(whitespace_email)
    vector_input = tfidf.transform([transformed_email])
    output = model.predict(vector_input)
    print(output)
    # return Response({"data":"Hi"})

    if output == [0]:
        return Response({"val": True})
    else:
        return Response({"val": False})