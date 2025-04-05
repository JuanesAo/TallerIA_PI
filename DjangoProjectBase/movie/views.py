from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import matplotlib.pyplot as plt  
import matplotlib  
import io  
import urllib, base64  
from.models import Movie
import numpy as np
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv
import os

import matplotlib.pyplot as plt
import matplotlib
import io
import urllib, base64

# Create your views here.
# Cargar la API Key de OpenAI
load_dotenv('../api_keys.env')
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Función para calcular similitud de coseno
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def home(request):  
    #return HttpResponse('<h1>Welcome to Home Page</h1>')
    #return render(request,'home.html')
    #return render(request, 'home.html', {'name': 'Maria Alejandra Ocampo Giraldo'})  
    searchTerm = request.GET.get('searchMovie')  
    
    if searchTerm:  
        movies = Movie.objects.filter(title__icontains=searchTerm)  
    else:  
        movies = Movie.objects.all()  
        
    return render(request, 'home.html', {'searchTerm': searchTerm, 'movies': movies})

                        
def about(request):
    #return HttpResponse('<h1>Welcome to About Page</h1>')
    return render(request, 'about.html', {'name': 'Marialita'})
def signup(request):  
    email = request.GET.get('email')  
    return render(request, 'signup.html', {'email': email})  

def statistics_view(request):
    # Obtener todas las películas
    all_movies = Movie.objects.all()

    # =================== GRÁFICA 1: PELÍCULAS POR AÑO ===================
    movie_counts_by_year = {}
    for movie in all_movies:
        year = str(movie.year) if movie.year else "None"
        movie_counts_by_year[year] = movie_counts_by_year.get(year, 0) + 1

    bar_width = 0.5
    bar_positions = range(len(movie_counts_by_year))

    plt.figure(figsize=(10, 5))
    plt.bar(bar_positions, movie_counts_by_year.values(), width=bar_width, align='center', color='red')
    plt.title('Movies per Year', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Movies', fontsize=12)
    plt.xticks(bar_positions, movie_counts_by_year.keys(), rotation=45)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')

    # =================== GRÁFICA 2: PELÍCULAS POR GÉNERO ===================
    genre_counts = {}
    for movie in all_movies:
        if movie.genre:  # Asegurar que hay un género registrado
            first_genre = movie.genre.split(',')[0].strip()  # Tomar solo el primer género
            genre_counts[first_genre] = genre_counts.get(first_genre, 0) + 1

    plt.figure(figsize=(10, 5))
    plt.bar(genre_counts.keys(), genre_counts.values(), color='blue')
    plt.title('Movies per Genre', fontsize=14)
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Number of Movies', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    image_png2 = buffer.getvalue()
    buffer.close()
    graphic2 = base64.b64encode(image_png2).decode('utf-8')

    # Renderizar la plantilla con ambas gráficas
    return render(request, 'statistics.html', {'graphic': graphic, 'graphic2': graphic2})

def recommend_movie(request):
    recommended_movie = None
    max_similarity = -1

    if request.method == "POST":
        # Recibir el prompt del usuario desde el formulario
        prompt = request.POST.get("prompt")

        # Generar embedding del prompt
        response = client.embeddings.create(
            input=[prompt],
            model="text-embedding-3-small"
        )
        prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)

        # Recorrer la base de datos y calcular similitudes
        for movie in Movie.objects.all():
            movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
            similarity = cosine_similarity(prompt_emb, movie_emb)

            if similarity > max_similarity:
                max_similarity = similarity
                recommended_movie = movie

    return render(request, "recommend_movie.html", {
        "recommended_movie": recommended_movie,
        "similarity": max_similarity,
    })