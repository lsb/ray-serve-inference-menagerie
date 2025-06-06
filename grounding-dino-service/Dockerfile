FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
      -f https://download.pytorch.org/whl/cu118/torch_stable.html && \
    pip install --no-cache-dir transformers==4.50.0 pillow==9.5.0 requests==2.31.0 \
      "ray[serve]==2.5.1" huggingface_hub==0.16.4

RUN git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e . && \
    cd .. && rm -rf sam2

RUN pip install fastapi uvicorn[standard]

COPY app.py /app.py

ENV PORT 8000
CMD ["python", "/app.py"]
