FROM python:3.12-slim

WORKDIR /app

# 0) System deps (curl optional)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# 1) Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Streamlit app
COPY app.py .

# 3) Render uses $PORT; local uses 8501 fallback
EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
