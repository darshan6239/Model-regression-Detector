FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Env vars expected at runtime:
#   GROQ_API_KEY       - required, free key from console.groq.com
#   SLACK_WEBHOOK_URL  - optional, skip Slack alerts if unset
#
# Example:
#   docker build -t model-regression-detector .
#   docker run --env-file .env model-regression-detector python main.py --prompt-version v1

ENTRYPOINT ["python", "main.py"]
CMD ["--prompt-version", "v1"]
