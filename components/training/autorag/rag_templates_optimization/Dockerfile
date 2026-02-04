

FROM redhat/ubi10-minimal:10.1@sha256:2910fd7720d45c3cabc708c71a6ff6b98695e1d814b71547c9324c97a728ebef

RUN microdnf -y install \
  python3 \
  python3-pip && \
  microdnf clean all

WORKDIR /usr/local/src/kfp/components

RUN --mount=type=bind,source=requirements.txt,target=/root/requirements.txt pip install -r /root/requirements.txt

COPY src/* ./
COPY component_metadata/* ./component_metadata/*

RUN cat <<EOF > kfp_config.ini 
[Components]
read_data = read_data_component.py
search_space_preparation = search_space_preparation_component.py
rag_templates_optimization = rag_templates_optimization_component.py
EOF

CMD ["/bin/bash", "-i"]
