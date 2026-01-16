import base64
from django.shortcuts import render
import requests

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