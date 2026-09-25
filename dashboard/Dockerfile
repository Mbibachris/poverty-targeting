# Container for the Streamlit dashboard, built by Hugging Face Spaces.
# The dashboard holds no model: it calls the prediction API over HTTPS.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py api_client.py ui_logic.py ./

# Hugging Face Spaces runs containers as user 1000.
RUN useradd --create-home --uid 1000 user
USER user

EXPOSE 8501
# The Space shows the app inside an iframe; without the two "false" settings, Streamlit's
# cross-site protections block file uploads there.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false", \
     "--browser.gatherUsageStats=false"]