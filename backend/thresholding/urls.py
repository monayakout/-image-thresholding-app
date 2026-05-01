from django.urls import path
from . import views

urlpatterns = [
    path('threshold/', views.ThresholdImageView.as_view(), name='threshold'),
    path('health/', views.health_check, name='health'),
]