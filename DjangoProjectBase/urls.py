from django.contrib import admin
from django.urls import path, include
from movie import views as movieViews

urlpatterns = [
    path('admin/', admin.site.urls),
    path('recommendation/', include('recommendation.urls')),  # Incluye las rutas de la app recommendation
    path("recommend/", movieViews.recommend_movie, name="recommend_movie"),
]
