import base64
from django.shortcuts import render
import requests

from classifier import models
from classifier.forms import ImageUploadForm
from classifier.models import Prediction

from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.core.files.storage import default_storage
from .ml_model import predict_from_image_path

def classify_image(request):
    result = None

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img = request.FILES['image']

            file_path = default_storage.save("tmp/" + img.name, img)
            absolute_path = default_storage.path(file_path)

            result = predict_from_image_path(absolute_path)

            Prediction.objects.create(
                image=img,
                prediction=result
            )
    else:
        form = ImageUploadForm()

    return render(request, "classify.html", {"form": form, "result": result})

def stats(request):
    predictions = Prediction.objects.all()
    total = predictions.count()

    if total > 0:
        most_common = predictions.values('prediction').annotate(
            count=models.Count('prediction')
        ).order_by('-count')[0]['prediction']
    else:
        most_common = None

    return render(request, "stats.html", {
        "total": total,
        "most_common": most_common
    })

@require_POST
def predict_view(request):
    image_file = request.FILES.get('image')
    if not image_file:
        return render(request, "index.html", {"error": "Nema slike."})

    file_path = default_storage.save("tmp/" + image_file.name, image_file)

    label = predict_from_image_path(file_path)

    return render(request, "result.html", {"label": label})
