FROM windpioneers/gdal-python:modest-heron-gdal-2.4.1-python-3.10-slim

WORKDIR /indices

COPY --chown=1000:1000 pyproject.toml /indices/pyproject.toml
COPY --chown=1000:1000 README.md /indices/README.md
COPY --chown=1000:1000 dask.yaml /indices/dask.yaml
COPY --chown=1000:1000 indices_compute /indices/indices_compute

RUN curl -sSL https://install.python-poetry.org | python3 

RUN /root/.local/bin/poetry install

ENTRYPOINT ["/indices/indices_compute"]
