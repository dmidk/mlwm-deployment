ARG BASE_IMAGE="pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime"
FROM ${BASE_IMAGE}

WORKDIR /workspace

COPY pyproject.toml .
COPY *.yaml ./
COPY entry.sh ./
COPY src/ ./src

# Download inference artifact from S3
ARG DEFAULT_ARTIFACT="s3://mlwm-artifacts/inference-artifacts/gefion-1.zip"
ENV MLWM_INFERENCE_ARTIFACT=${DEFAULT_ARTIFACT}

RUN apt-get update && apt-get install -y python3 python3-pip unzip git
RUN pip3 install awscli

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

# Get inference artifact from S3
RUN aws s3 cp $MLWM_INFERENCE_ARTIFACT ./inference_artifact.zip
RUN mkdir -p /workspace/inference_artifact
RUN unzip inference_artifact.zip -d /workspace/inference_artifact
RUN rm inference_artifact.zip
# List files in the inference artifact directory for verification
RUN ls -la /workspace/inference_artifact

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# export current torch version to constraints file
RUN python -c 'import torch; print(f"torch=={torch.__version__}")' > constraints.txt
# install with constraints
RUN uv pip install --break-system-packages --system --constraints constraints.txt .
# check that we can print out neural-lam version
RUN python -c 'import neural_lam; print(f"neural-lam=={neural_lam.__version__}")'

# inside the container we have installed directly into system python, so we
# wont use uv here (otherwise uv tries to create a new virtual environment)
ENV USE_UV="false"
# Set the default command to run when the container starts
ENTRYPOINT ["./entry.sh"]
# No arguments passed by default
CMD [""]
