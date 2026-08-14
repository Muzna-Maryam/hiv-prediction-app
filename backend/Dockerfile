# Pinned to a specific Python version rather than "whatever's on my
# machine" (3.14 locally) - the whole point of a container is a fixed,
# reproducible runtime regardless of the host. 3.12-slim also has broad
# prebuilt-wheel coverage for shap's C extensions, which a bleeding-edge
# Python version risks not having yet, forcing a slow from-source
# compile in an image with no build tools installed.
#
# Model artifacts are deliberately NOT copied into this image - see the
# volume mounts in docker-compose.yml. This image is code + deps only,
# so rebuilding it is only needed when the code changes, never when you
# retrain the model.
FROM python:3.12-slim

WORKDIR /app

# requirements copied and installed before the rest of the code, so
# Docker's layer cache reuses the (slow) pip install step across builds
# where only app code changed, not dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["./entrypoint.sh"]