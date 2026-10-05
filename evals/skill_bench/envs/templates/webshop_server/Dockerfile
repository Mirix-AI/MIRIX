# WebShop server image.
# Built by `python -m evals.skill_bench.envs.webshop --build`, for linux/amd64 only. The base
# image is pinned by digest and must equal protocol.WEBSHOP_SERVER["base_image"]; the tag of
# the built image is derived from the files of this directory, so any change here yields a
# new image. The image holds upstream princeton-nlp/WebShop at a pinned commit, the full
# product set (Hugging Face mirror pinned by revision and sha256, data.json) and its Lucene
# index built with --threads 1; build.json records the commit, the data hashes, the hash of
# the index record (a deterministic description of the index, data_index.py), the spaCy
# model, the fuzzy backend and the build context's digest, which server.py reports on
# /ready and the harness checks before a cell starts (the index, data and commit also
# against the frozen manifest's source.json).
#
# Layer order (build cache): what is slow and changes rarely comes first. The product files
# (about 5.7 GB) depend only on data.json and data_index.py; the index on those, the pinned
# packages and the upstream commit; server.py, its install check, the build context's
# digest and the build record come last, so a change to server.py rebuilds only them and
# never downloads the data or rebuilds the index again (tests/test_env_webshop.py pins this
# order).
FROM python:3.9-bullseye@sha256:3dcaf39bbe2d6d8031a4a7ce68a20f2d494419aca18ec064f9e82e146e97a0ed

# PYTHONHASHSEED: upstream draws its prices over lists, but no set or dict of strings may
# ever order a random draw differently between two server starts.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TQDM_DISABLE=1 \
    JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    WEBSHOP_DIR=/opt/webshop \
    WEBSHOP_DATA_DIR=/opt/webshop-data \
    WEBSHOP_BUILD_RECORD=/opt/skill-bench/build.json

# The pinned product files (about 5.7 GB), verified by size and sha256, outside the
# upstream checkout. Standard library only: nothing installed below can invalidate them.
COPY data.json data_index.py /opt/skill-bench/
RUN python /opt/skill-bench/data_index.py fetch-data --out /opt/webshop-data

# Java 11 for pyserini 0.17 / Lucene 8 (upstream setup.sh:29); bullseye ships openjdk-11.
# apt reads snapshot.debian.org at the base image's own date: the image's sources.list
# names exactly these three suites at this date, commented out. Debian bullseye's LTS has
# ended, so its regular mirrors may no longer serve its packages; the snapshot never
# changes, so the Java runtime and its libraries are fixed with the base digest, and they
# match what the image already has installed (nothing is upgraded). build.json still
# records the Java runtime.
RUN printf '%s\n' \
        'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20250721T000000Z bullseye main' \
        'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/20250721T000000Z bullseye-security main' \
        'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20250721T000000Z bullseye-updates main' \
        > /etc/apt/sources.list \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        openjdk-11-jdk-headless git \
    && rm -rf /var/lib/apt/lists/*

# The hash-locked allowlist only (never upstream's requirements.txt). gym 0.24.0 is an
# sdist: it builds with the base image's setuptools (pinned with the image digest).
COPY requirements.txt /opt/skill-bench/requirements.txt
RUN pip install --no-cache-dir --disable-pip-version-check --no-deps --require-hashes \
        --no-build-isolation -r /opt/skill-bench/requirements.txt \
    && rm -rf /root/.cache

# Upstream at the pinned commit, unchanged. Declared here: only the layers from the clone
# on depend on it.
ARG WEBSHOP_COMMIT=64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
RUN git clone https://github.com/princeton-nlp/WebShop.git /opt/webshop \
    && git -C /opt/webshop checkout --detach "$WEBSHOP_COMMIT" \
    && test "$(git -C /opt/webshop rev-parse HEAD)" = "$WEBSHOP_COMMIT"

# The full index only, --threads 1 (upstream run_indexing.sh), and its record: the
# documents' sha256 and number, the indexer's arguments, pyserini's version and the
# index's statistics (the same bytes on every build; the index files' own hashes differ
# on every build and are kept for diagnosis only).
RUN python /opt/skill-bench/data_index.py build-index --data-dir /opt/webshop-data

# Everything below changes more often; none of it fetches or indexes anything.
COPY server.py /opt/skill-bench/
# The spaCy model and the fuzzy backend upstream's reward runs with, and every upstream
# module the server imports must import here (the JVM pyserini needs included), not
# first at serve time.
RUN python /opt/skill-bench/server.py check-install
# The build context's digest (envs/webshop.py passes it; the image's tag): /ready reports
# it and the harness refuses a server built from another context. Declared here so that
# only the record's layer depends on it. build.json also keeps the Java runtime,
# setuptools and a digest of every installed package, which no pin fixes.
ARG SKILL_BENCH_CONTEXT
RUN python /opt/skill-bench/server.py write-build-record --data-dir /opt/webshop-data

WORKDIR /opt/webshop
EXPOSE 8000
ENTRYPOINT ["python", "/opt/skill-bench/server.py"]
# The protocol seed; /manifest is envs/splits/webshop mounted read-only.
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--seed", "42", "--manifest-dir", "/manifest"]
