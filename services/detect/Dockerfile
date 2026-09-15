FROM python:3.11-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
RUN groupadd -r sentinel && useradd -m -r -g sentinel sentinel
ENV HF_HOME=/tmp
ENV TORCH_HOME=/tmp
COPY requirements.txt .

# CPU-only torch, installed FIRST from the PyTorch CPU index. Installing it
# before requirements.txt means the `torch==2.3.0` line below is already
# satisfied (2.3.0+cpu matches ==2.3.0) and pip never fetches the CUDA build.
# The default linux/amd64 wheel on PyPI is the CUDA 12.1 build, which drags in
# ~2.5 GB of nvidia-* packages for a box that has no GPU.
# See docs/DEPLOYMENT_FREE_TIER.md §5.2.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.0 torchvision==0.18.0 \
 && pip install --no-cache-dir -r requirements.txt
COPY app/ app/
RUN chown -R sentinel:sentinel /app
USER sentinel
EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
