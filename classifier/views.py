from django.utils import timezone
from datetime import timedelta
from django.shortcuts import render

from classifier import models
from classifier.forms import ImageUploadForm
from classifier.models import Prediction

from django.db.models import Count
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.core.files.storage import default_storage
from .ml_model import predict_from_image_path

LABEL_MAP = {
    "01-minor": "Minor",
    "02-moderate": "Moderate",
    "03-severe": "Severe"
}

def classify_image(request):
    result = None
    image_url = None

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img = request.FILES['image']

            file_path = default_storage.save("tmp/" + img.name, img)
            absolute_path = default_storage.path(file_path)

            result = predict_from_image_path(absolute_path)

            image_url = default_storage.url(file_path)

            Prediction.objects.create(
                image=img,
                prediction=result
            )
    else:
        form = ImageUploadForm()

    return render(request, "classify.html", {"form": form, "result": result, "image_url": image_url})

def stats(request):
    predictions = Prediction.objects.all()

    if predictions.exists():
        most_common_raw = predictions.values('prediction').annotate(
            count=Count('prediction')
        ).order_by('-count')[0]['prediction']
        most_common = LABEL_MAP.get(most_common_raw, most_common_raw)
    else:
        most_common = None

    last_24h = timezone.now() - timedelta(hours=24)
    recent_count = predictions.filter(created_at__gte=last_24h).count()

    raw_counts = predictions.values('prediction').annotate(count=Count('prediction'))
    class_distribution = {
        LABEL_MAP.get(item['prediction'], item['prediction']): item['count']
        for item in raw_counts
    }

    return render(request, "stats.html", {
        "most_common": most_common,
        "recent_count": recent_count,
        "class_distribution": class_distribution,
    })

@require_POST
def predict_view(request):
    image_file = request.FILES.get('image')
    if not image_file:
        return render(request, "index.html", {"error": "Nema slike."})

    file_path = default_storage.save("tmp/" + image_file.name, image_file)

    label = predict_from_image_path(file_path)

    return render(request, "result.html", {"label": label})
