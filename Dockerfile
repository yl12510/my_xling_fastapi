FROM tiangolo/uvicorn-gunicorn-fastapi:python3.6 

EXPOSE 80 

LABEL description="XLING as a service on top FastAPI" 

# Install framework 
RUN pip install tensorflow==1.12.0 tensorflow-hub tf-sentencepiece 

COPY ./app /app 
