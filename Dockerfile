# FarmScan on Railway. A Dockerfile (not Railpack) so the exact runtime system
# libraries the binary wheels need are present: libexpat1 for rasterio/GDAL,
# libgomp1 for scikit-learn/scipy's OpenMP. The app lives in backend/.
FROM python:3.13-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# PORT is set by Railway; default 8000 for a local run.
CMD ["sh", "-c", "uvicorn main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000}"]
