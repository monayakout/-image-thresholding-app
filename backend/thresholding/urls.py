from django.urls import path

from segmentation.views import SegmentationImageView

from . import views

urlpatterns = [
    path("threshold/", views.ThresholdImageView.as_view(), name="threshold"),
    path("health/", views.health_check, name="health"),
    path(
        "segment/",
        SegmentationImageView.as_view(),
        name="segment",
    ),
    path(
        "segment/<str:method_slug>/",
        SegmentationImageView.as_view(),
        name="segment-method",
    ),
]