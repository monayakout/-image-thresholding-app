from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .segmentation_controller import run_segmentation


class SegmentationImageView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, method_slug=None, *args, **kwargs):
        if "image" not in request.FILES:
            return Response(
                {"error": 'No image file provided. Use form key "image".'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        image_file = request.FILES["image"]
        data = request.data.copy()
        for key in request.query_params:
            if key not in data:
                data[key] = request.query_params[key]

        slug = (
            method_slug
            or data.get("method")
            or request.query_params.get("method")
            or "kmeans"
        )

        try:
            result = run_segmentation(str(slug), image_file, data)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(
                {"error": f"Segmentation failed: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(result, status=status.HTTP_200_OK)
