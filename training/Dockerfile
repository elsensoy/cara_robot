# training/Dockerfile
FROM nvcr.io/nvidia/pytorch:24.06-py3
RUN apt-get update && apt-get install -y python3-opencv \
    && pip install --no-cache-dir transformers==4.44.2 timm==1.0.9 albumentations
WORKDIR /workspace
COPY . /workspace
CMD ["python", "train_personal_vit.py"]
