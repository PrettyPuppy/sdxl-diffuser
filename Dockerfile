FROM alpine/git:2.36.2 as download
RUN git clone YOUR_DOCKER_REPOSITORY
RUN cd DIRECTORY
RUN rm -rf .git
FROM python:3.10.9-slim as build_final_image
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    ROOT=/sdxl-main \
    PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# RUN --mount=type=cache,target=/cache --mount=type=cache,target=/root/.cache/pip \
#     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY requirements.txt /requirements.txt
COPY start.sh /start.sh
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt
RUN chmod +x /start.sh
CMD /start.sh