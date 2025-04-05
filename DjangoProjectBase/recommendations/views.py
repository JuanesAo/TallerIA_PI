from django.shortcuts import render

# Create your views here.
import numpy as np
import os
from dotenv import load_dotenv
from django.shortcuts import render
from openai import OpenAI
from movie.models import Movie

# Cargar la API Key
load_dotenv('./api_keys.env')
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Función para calcular similitud de coseno
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recommend_movie(request):
    recommended_movie = None
    prompt = ""
    if request.method == 'POST':
        prompt = request.POST.get('prompt')
        # Obtener embedding del prompt
        response = client.embeddings.create(
            input=[prompt],
            model="text-embedding-3-small"
        )
        prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)

        # Buscar la película más parecida
        max_similarity = -1
        for movie in Movie.objects.all():
            movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
            similarity = cosine_similarity(prompt_emb, movie_emb)
            if similarity > max_similarity:
                max_similarity = similarity
                recommended_movie = movie

    return render(request, 'recommendations/recommend.html', {
        'prompt': prompt,
        'movie': recommended_movie
    })