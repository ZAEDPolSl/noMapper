FROM python:3.8-slim

# metadata
LABEL software="noMapper"
LABEL software.description="A selector separating reads that do not belong to the gene being searched from those that may belong to it."
LABEL author="Tomasz Strzoda"
LABEL author.email="tomasz.strzoda@polsl.pl"

# scripts
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY ./app /app

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]