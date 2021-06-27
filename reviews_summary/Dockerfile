FROM nvcr.io/nvidia/pytorch:19.01-py3

RUN mkdir -p /reviews_summary
WORKDIR /reviews_summary

COPY . /reviews_summary
WORKDIR /reviews_summary

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U pip setuptools wheel
RUN pip install -U spacy==2.3.7
RUN python -m spacy download en_core_web_sm

CMD ["/bin/bash"]
