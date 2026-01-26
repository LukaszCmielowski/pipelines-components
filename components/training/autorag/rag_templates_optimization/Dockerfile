

FROM redhat/ubi10-minimal:10.1@sha256:2910fd7720d45c3cabc708c71a6ff6b98695e1d814b71547c9324c97a728ebef

RUN microdnf install \
  python3

COPY requirements.txt $HOME

RUN pip install -r requirements.txt
# ai4rag, docling 

CMD ["/bin/bash", "-i"]
