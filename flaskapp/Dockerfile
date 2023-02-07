FROM python:3.10-alpine

# copy every content from the local file to the image
COPY . /HomeWork1

# switch working directory
WORKDIR /HomeWork1

# install the dependencies and packages in the requirements file
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["main.py" ]