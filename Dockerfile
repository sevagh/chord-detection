FROM python

RUN apt-get update -y && apt-get install -y \
	libsndfile-dev

WORKDIR /chord-detection
COPY . .

RUN pip install -r ./requirements.txt

ENTRYPOINT ["python3.7", "multipitch.py"]
