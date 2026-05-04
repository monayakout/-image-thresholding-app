from django.urls import path
from .views import PredictFaceView

urlpatterns = [
    path('predict/', PredictFaceView.as_view(), name='predict-face'),
]
