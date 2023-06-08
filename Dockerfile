# Create a docker file for python 3.10 from an alpine image
# Use pip to install flask
# Use pip to install tiktoken_util.py
# Copy the /usr/tiktoken_util.py/app_server.py file to the container
# The entry point for the container is python -m flask --app /usr/tiktoken_util.py/app_server.py run

FROM python:3.10-alpine

RUN pip install --upgrade pip
RUN pip install tiktoken flask pypdfium2 langchain
ADD src /app/

CMD ["python", "-m", "flask", "--app", "/app/app_server.py", "run", "--host", "0.0.0.0"]

EXPOSE 5000

