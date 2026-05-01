from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser, FormParser
from .processing import run_all_thresholds


class ThresholdImageView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response(
                {'error': 'No image file provided. Use key "image".'},
                status=status.HTTP_400_BAD_REQUEST
            )

        image_file = request.FILES['image']

        try:
            spectral_classes = int(request.data.get('spectral_classes', 3))
            local_block      = int(request.data.get('local_block', 35))
            local_offset     = float(request.data.get('local_offset', 10))
        except (ValueError, TypeError) as e:
            return Response({'error': f'Invalid parameter: {str(e)}'}, status=400)

        try:
            result = run_all_thresholds(
                image_file,
                spectral_classes=spectral_classes,
                local_block=local_block,
                local_offset=local_offset,
            )
        except Exception as e:
            return Response({'error': f'Processing failed: {str(e)}'}, status=500)

        return Response(result, status=status.HTTP_200_OK)


@api_view(['GET'])
def health_check(request):
    return Response({'status': 'ok', 'message': 'API is running.'})