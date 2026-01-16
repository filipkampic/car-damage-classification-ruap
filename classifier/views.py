import base64
from django.shortcuts import render
import requests

from classifier import models
from classifier.forms import ImageUploadForm
from classifier.models import Prediction

AZURE_ENDPOINT = "<YOUR_ENDPOINT_URL>"
AZURE_KEY = "<YOUR_ENDPOINT_KEY>"

def classify_image(request):
    result = None

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img = request.FILES['image']

            b64 = base64.b64encode(img.read()).decode('utf-8')

            payload = {"image_base64": b64}
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AZURE_KEY}"
            }

            response = requests.post(AZURE_ENDPOINT, json=payload, headers=headers)
            result = response.json().get("prediction", "error")

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