FROM continuumio/miniconda3:22.11.1

RUN apt-get update; apt-get upgrade -y;
RUN apt install -y build-essential zlib1g-dev libbz2-dev liblzma-dev sudo;  

COPY environment.yml /tmp/environment.yml
RUN conda env update -f /tmp/environment.yml

RUN echo "conda activate peps" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /app

COPY assets/ /app/assets/
COPY Web_app/ /app/Web_app/

WORKDIR /app/Web_app

ENV PORT 8501
RUN bash setup.sh
COPY entrypoint.sh /usr/bin/entrypoint.sh
RUN chmod +x /usr/bin/entrypoint.sh

ENTRYPOINT ["/usr/bin/entrypoint.sh"]
