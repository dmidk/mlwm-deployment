INFERENCE_ARTIFACT_URI="s3://mlwm-artifacts/inference-artifacts/surface-dummy-model.zip"

# 1. copy inference artifact from s3 bucket
aws s3 cp ${INFERENCE_ARTIFACT_URI} ./inference_artifact.zip

# 2. unzip inference artifact to (./inference_artifact)
unzip inference_artifact.zip -d ./inference_artifact

# 2. build image (this should copy the inference artifact to the image and unpack it)
# docker build -t surface-dummy-model_DINI \
#     --build-arg INFERENCE_ARTIFACT_PATH=./inference_artifact \
#     -f Dockerfile .

# 3. clean up
# rm inference_artifact.zip
# rm -rf inference_artifact/
