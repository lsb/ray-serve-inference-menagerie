FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2+cu118 \
      -f https://download.pytorch.org/whl/cu118/torch_stable.html && \
    pip install --no-cache-dir transformers==4.52.4 pillow==9.5.0 requests==2.31.0 \
      "ray[serve]==2.46.0"

COPY app.py /app.py

ENV PORT 8000
CMD ["python", "/app.py"]
