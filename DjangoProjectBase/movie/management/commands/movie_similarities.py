import os
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv

class Command(BaseCommand):
    help = "Compare multiple movie pairs and prompts using OpenAI embeddings"

    def handle(self, *args, **kwargs):
        load_dotenv('../api_keys.env')
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

        
        comparisons = [
            {
                "movie1": "Frankenstein",
                "movie2": "The Great Train Robbery",
                "prompt": "película clásica de terror"
            },
            {
                "movie1": "The Life and Death of King Richard III",
                "movie2": "Cabiria",
                "prompt": "película histórica"
            },
            {
                "movie1": "The Avenging Conscience or Thou Shalt Not Kill",
                "movie2": "Oppenheimer",
                "prompt": "película sobre culpa y moralidad"
            }
        ]

        def get_embedding(text):
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return np.array(response.data[0].embedding, dtype=np.float32)

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        for combo in comparisons:
            try:
                movie1 = Movie.objects.get(title=combo["movie1"])
                movie2 = Movie.objects.get(title=combo["movie2"])
            except Movie.DoesNotExist:
                self.stdout.write(f"❗ No encontré una de las películas: '{combo['movie1']}' o '{combo['movie2']}'")
                continue

            emb1 = get_embedding(movie1.description)
            emb2 = get_embedding(movie2.description)
            similarity = cosine_similarity(emb1, emb2)

            prompt_emb = get_embedding(combo["prompt"])
            sim_prompt_movie1 = cosine_similarity(prompt_emb, emb1)
            sim_prompt_movie2 = cosine_similarity(prompt_emb, emb2)

            self.stdout.write(f"\n🎬 Comparación: '{movie1.title}' vs '{movie2.title}'")
            self.stdout.write(f"📊 Similaridad entre películas: {similarity:.4f}")
            self.stdout.write(f"📝 Similaridad prompt vs '{movie1.title}': {sim_prompt_movie1:.4f}")
            self.stdout.write(f"📝 Similaridad prompt vs '{movie2.title}': {sim_prompt_movie2:.4f}")