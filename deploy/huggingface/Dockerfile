# The Space runs the exact image CI published - not a rebuild of it.
#
# Building from source here would create a second artifact that could drift
# from the one that passed the smoke test. Pulling the published digest means
# the live demo and the release are the same bytes.
FROM ghcr.io/yahya-osama-mohmamed/insurance-fraud-api:latest

# Hugging Face Spaces routes to 7860.
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
