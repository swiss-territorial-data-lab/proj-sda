FROM object-detector-stdl-objdet

# Work in a docker folder and not in /app like the OD
WORKDIR /docker_mount

USER root:root
RUN apt update && apt install -y git && apt clean
RUN apt-get install unzip

ADD requirements.txt proj-sda/requirements.txt
RUN pip install -r proj-sda/requirements.txt

CMD /bin/bash