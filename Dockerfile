FROM object-detector-stdl-objdet

# Work in a docker folder and not in /app like the OD
WORKDIR /docker_mount

USER root:root
RUN apt update && apt install -y git && apt clean

ADD requirements.txt proj-borderpoints/requirements.txt
RUN pip install -r proj-borderpoints/requirements.txt