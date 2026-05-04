from django.urls import path
from .views import PredictFaceView, DetectFaceView

urlpatterns = [
    path('detect/', DetectFaceView.as_view(), name='detect-face'),
    path('predict/', PredictFaceView.as_view(), name='predict-face'),
]
