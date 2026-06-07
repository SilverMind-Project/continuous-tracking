# Deploy the INT8 model repository to Jetson Orin Nano Super

This runbook moves the Continuous Tracking System (CTS) INT8 model repository
to an NVIDIA Jetson Orin Nano Super, builds TensorRT plans on that Jetson, starts
Triton Inference Server, and points CTS and the person-identification service at
the remote endpoint.

The procedure uses placeholders throughout. Do not commit private hostnames,
addresses, usernames, calibration images, credentials, or household data.

## Deployment status and scope

The Jetson repository contains eight explicit-Q/DQ ONNX graphs:

| Triton model | Purpose | Input contract |
| --- | --- | --- |
| `person-detector` | YOLO26L person detection | Fixed `[8,3,640,640]` |
| `pose-rtmpose` | RTMPose-m pose estimation | Dynamic batch 1 to 8 |
| `reid-solider` | SOLIDER body re-identification | Dynamic batch 1 to 8 |
| `face-detector-scrfd` | Buffalo_L face detection | Fixed batch 1 |
| `face-recognition-arcface` | Buffalo_L face recognition | Fixed batch 1 |
| `face-landmark-2d106` | Buffalo_L 2D landmarks | Fixed batch 1 |
| `face-landmark-3d68` | Buffalo_L 3D landmarks | Fixed batch 1 |
| `face-attribute-genderage` | Buffalo_L gender and age | Fixed batch 1 |

The Q/DQ ONNX files have passed model-specific output-regression checks. The
TensorRT plans remain target-specific and must be built on the Jetson that will
serve them.

Depth Anything V2 is intentionally excluded.

## Understand the artifacts

Three artifact classes have different handling requirements:

| Artifact | Portable | Commit to Git | Notes |
| --- | --- | --- | --- |
| `model_int8.onnx` | Yes, subject to runtime compatibility | Git LFS only, if licensing permits | Source artifact for TensorRT builds |
| `model.plan` | No | No | Tied to GPU, JetPack, CUDA, and TensorRT versions |
| Calibration images and tensors | No | No | Contains identifiable household data |

Treat the validated Q/DQ ONNX files as the durable deployment source. Rebuild
TensorRT plans after a JetPack, CUDA, TensorRT, or target hardware change.

## Prerequisites

### Workstation

- a complete checkout of `continuous-tracking`;
- all eight `triton-models-jetson/*/1/model_int8.onnx` files;
- SSH access to the Jetson;
- `rsync`, `sha256sum`, Git, and Git LFS.

### Jetson

- Jetson Orin Nano Super with 8 GB unified memory;
- Jetson Linux 36.x and TensorRT 10.3 or a separately qualified combination;
- Docker, Docker Compose, and NVIDIA Container Runtime;
- active cooling and adequate storage;
- SSH enabled before switching to headless boot.

TensorRT engine construction needs substantially more free and contiguous
unified memory than serving the completed plans.

## 1. Define deployment variables

Run these commands on the workstation:

```bash
cd continuous-tracking

export JETSON_HOST=jetson-hostname-or-ip
export JETSON_USER=jetson-user
export JETSON_TARGET="${JETSON_USER}@${JETSON_HOST}"
export JETSON_DIR=continuous-tracking
```

Test SSH:

```bash
ssh "$JETSON_TARGET" 'hostname; uname -m'
```

The expected architecture is `aarch64`.

## 2. Verify the local model set

List the Q/DQ models:

```bash
find triton-models-jetson \
  -type f \
  -name model_int8.onnx \
  -exec ls -lh {} \;
```

Require exactly eight:

```bash
test "$(find triton-models-jetson -type f -name model_int8.onnx | wc -l)" -eq 8
```

Create a transfer manifest:

```bash
find triton-models-jetson \
  -type f \
  -name model_int8.onnx \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > jetson-models.sha256
```

## 3. Copy the repository to the Jetson

Use this path while model files are local or are not published through Git LFS.

Create the remote directory:

```bash
ssh "$JETSON_TARGET" "mkdir -p ~/${JETSON_DIR}/triton-models-jetson"
```

Copy the Jetson model repository:

```bash
rsync -av --info=progress2 \
  --exclude='reports/' \
  --exclude='**/__pycache__/' \
  --exclude='**/model.plan' \
  triton-models-jetson/ \
  "${JETSON_TARGET}:~/${JETSON_DIR}/triton-models-jetson/"
```

Copy the deployment Compose file and checksum manifest:

```bash
rsync -av \
  docker-compose.jetson-triton.yml \
  jetson-models.sha256 \
  "${JETSON_TARGET}:~/${JETSON_DIR}/"
```

Verify every copied model:

```bash
ssh "$JETSON_TARGET" "
  cd ~/${JETSON_DIR} &&
  sha256sum -c jetson-models.sha256
"
```

Do not proceed if a checksum fails.

## 4. Inspect the Jetson software stack

Connect to the Jetson:

```bash
ssh "$JETSON_TARGET"
cd ~/continuous-tracking
```

Inspect Jetson Linux and inference packages:

```bash
cat /etc/nv_tegra_release

dpkg-query -W \
  nvidia-l4t-core \
  nvidia-l4t-cuda \
  nvidia-l4t-3d-core \
  tensorrt
```

The `nvidia-jetpack` metapackage may be absent even when the required L4T,
CUDA, and TensorRT packages are installed. Verify the component packages
instead of treating a missing metapackage as a driver failure.

Check Docker:

```bash
docker --version
docker compose version
docker info | sed -n '/Runtimes/,+3p'
```

Do not install desktop `nvidia-driver-*` packages on Jetson. The GPU driver is
part of the Jetson Linux BSP and must be updated through the supported JetPack
or Jetson Linux path.

## 5. Select and test the Triton iGPU image

The build and server must use the same JetPack-compatible Triton iGPU image:

```bash
export TRITON_JETSON_IMAGE=nvcr.io/nvidia/tritonserver:24.08-py3-igpu
docker pull "$TRITON_JETSON_IMAGE"
```

Confirm `trtexec` is present:

```bash
docker run --rm --runtime nvidia \
  "$TRITON_JETSON_IMAGE" \
  bash -lc \
  'command -v trtexec || test -x /usr/src/tensorrt/bin/trtexec'
```

### Jetson BSP driver requirement warning

Some Triton iGPU images apply a discrete-GPU driver requirement before
launching the requested command. A Jetson Linux 36.x system may report BSP
driver `540.4.0`, while the container requests desktop driver `560.28` or
later.

Do not assume the warning is harmless. First run a small TensorRT build with an
explicit, opt-in requirement override:

```bash
export NVIDIA_DISABLE_REQUIRE=1

docker run --rm --runtime nvidia \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -v "$PWD/triton-models-jetson:/models:rw" \
  "$TRITON_JETSON_IMAGE" \
  bash -lc '/usr/src/tensorrt/bin/trtexec \
    --onnx=/models/face-attribute-genderage/1/model_int8.onnx \
    --saveEngine=/models/face-attribute-genderage/1/test.plan \
    --minShapes=data:1x3x96x96 \
    --optShapes=data:1x3x96x96 \
    --maxShapes=data:1x3x96x96 \
    --skipInference \
    --stronglyTyped'
```

Proceed only when the output confirms all of the following:

- the selected device is `Orin`;
- the expected TensorRT version is loaded;
- CUDA initializes successfully;
- the engine is written;
- `trtexec` ends with `PASSED`.

Remove the temporary engine:

```bash
rm -f triton-models-jetson/face-attribute-genderage/1/test.plan
```

If the test fails before detecting the Orin GPU, stop and select a Triton image
compatible with the installed Jetson stack. Do not use the requirement override
to conceal a real CUDA or TensorRT incompatibility.

## 6. Prepare unified memory for engine construction

Jetson uses unified CPU and GPU memory. Linux `available` memory alone is not a
sufficient build signal. TensorRT tactics can require large free blocks, shown
as `lfb` by `tegrastats`.

Stop running containers without passing an empty argument list to
`docker stop`:

```bash
containers="$(docker ps -q)"
if [ -n "$containers" ]; then
  docker stop $containers
fi
```

Stop other inference services if present:

```bash
systemctl is-active ollama >/dev/null 2>&1 &&
  sudo systemctl stop ollama
```

When connected over SSH, stop the graphical desktop:

```bash
sudo systemctl stop display-manager
```

Inspect memory and fragmentation:

```bash
free -h
swapon --show
ps -eo pid,user,rss,comm,args --sort=-rss | head -n 20
sudo fuser -v /dev/nvhost-gpu
timeout 3s tegrastats
```

An output such as `lfb 23x4MB` means only about 92 MB is available in large
free blocks even when `free -h` reports several gigabytes available. TensorRT
may then skip every detector tactic with messages such as:

```text
Tactic Device request: 150MB Available: 107MB
Could not find any implementation for node ...
```

In this case, the apparent operator error is secondary. Every implementation
was rejected because no sufficiently large memory block was available.

Try memory compaction:

```bash
sudo sh -c 'sync; echo 1 > /proc/sys/vm/compact_memory'
timeout 3s tegrastats
```

If the large free block count remains low, boot headless and build immediately
after reboot:

```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

Reconnect over SSH:

```bash
ssh "$JETSON_TARGET"
cd ~/continuous-tracking
free -h
timeout 3s tegrastats
```

Do not use swap as the primary fix. Swap can protect CPU processes from an
out-of-memory kill, but it does not create large GPU-addressable physical
memory blocks for TensorRT tactics.

## 7. Build all TensorRT plans

Set the qualified image and conservative 8 GB build profile:

```bash
cd ~/continuous-tracking

export TRITON_JETSON_IMAGE=nvcr.io/nvidia/tritonserver:24.08-py3-igpu
export NVIDIA_DISABLE_REQUIRE=1
export TRT_WORKSPACE_MB=256
export TRT_BUILDER_OPT_LEVEL=2
export TRT_MAX_AUX_STREAMS=0
export TRT_SPARSITY=disable
```

Build:

```bash
chmod +x triton-models-jetson/scripts/build_tensorrt_plans.sh
rm -f triton-models-jetson/*/1/model.plan
bash triton-models-jetson/scripts/build_tensorrt_plans.sh
```

The fixed batch-1 detector is built first. Do not copy the older batch-8 Q/DQ
graph to an 8 GB Jetson: TensorRT can fail after tactic selection because no
implementation fits the remaining builder memory. Each successful model ends
with:

```text
PASSED TensorRT.trtexec
```

If the batch-1 detector still fails after a clean headless reboot, retry once
with minimal tactic exploration:

```bash
export TRT_BUILDER_OPT_LEVEL=0
bash triton-models-jetson/scripts/build_tensorrt_plans.sh
```

This can reduce engine build pressure, but it may also produce a slower engine.
Benchmark the result before production use.

## 8. Verify plans and INT8 layer reports

Require eight plans:

```bash
find triton-models-jetson -type f -name model.plan -exec ls -lh {} \;
test "$(find triton-models-jetson -type f -name model.plan | wc -l)" -eq 8
```

Inspect generated reports:

```bash
find triton-models-jetson/reports -maxdepth 1 -type f -printf '%f\n' | sort
```

The build script runs `verify_tensorrt_precision.py` after all builds. It fails
if any layer report contains no INT8 tensor format.

Record the target software stack and model checksums:

```bash
mkdir -p triton-models-jetson/reports

cat /etc/nv_tegra_release \
  > triton-models-jetson/reports/jetson-linux.txt

dpkg-query -W \
  nvidia-l4t-core \
  nvidia-l4t-cuda \
  nvidia-l4t-3d-core \
  tensorrt \
  > triton-models-jetson/reports/runtime-packages.txt

find triton-models-jetson \
  -type f \
  \( -name model_int8.onnx -o -name model.plan \) \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > triton-models-jetson/reports/model-checksums.sha256
```

Reports are ignored by Git. Copy them to private operational storage if they
are needed for an audit.

## 9. Start Triton

Keep the image and explicit compatibility setting in the shell:

```bash
export TRITON_JETSON_IMAGE=nvcr.io/nvidia/tritonserver:24.08-py3-igpu
export NVIDIA_DISABLE_REQUIRE=1
```

Validate the rendered Compose configuration:

```bash
docker compose -f docker-compose.jetson-triton.yml config
```

Start Triton:

```bash
docker compose -f docker-compose.jetson-triton.yml up -d
docker logs -f cts-triton-jetson
```

Stop following logs with `Ctrl+C`. The container continues running.

## 10. Verify Triton readiness

Verify server readiness on the Jetson:

```bash
curl -fsS http://localhost:8700/v2/health/ready
echo
```

Verify every model:

```bash
for model in \
  person-detector \
  pose-rtmpose \
  reid-solider \
  face-detector-scrfd \
  face-recognition-arcface \
  face-landmark-2d106 \
  face-landmark-3d68 \
  face-attribute-genderage
do
  curl -fsS "http://localhost:8700/v2/models/${model}/ready"
  echo " ${model}: READY"
done
```

From the client host:

```bash
export JETSON_HOST=jetson-hostname-or-ip

curl -fsS "http://${JETSON_HOST}:8700/v2/health/ready"
curl -fsS "http://${JETSON_HOST}:8700/v2/models/person-detector/ready"
```

Triton exposes:

| Port | Protocol | Purpose |
| ---: | --- | --- |
| `8700` | HTTP | Health, metadata, and HTTP inference |
| `8701` | gRPC | CTS and person-identification inference |
| `8702` | HTTP | Prometheus metrics |

Keep these ports on the trusted LAN. Do not expose Triton directly to the
public Internet.

## 11. Point person identification at the Jetson

On the host that runs `person-identification-service`:

```bash
export JETSON_HOST=jetson-hostname-or-ip
export TRITON_GRPC_URL="${JETSON_HOST}:8701"
export PERSON_ID_MODEL_PROFILE=int8

docker compose up -d --build person-id
docker logs -f person-id
```

The service has one Triton inference code path. Startup fails unless all five
Buffalo_L models are ready. There is no local ONNX Runtime fallback or
automatic failover to another endpoint.

Verify:

```bash
curl -fsS http://localhost:8200/health
```

Confirm the response reports:

```text
inference_backend: triton
model_profile: int8
triton_endpoint: jetson-hostname-or-ip:8701
```

## 12. Point the tracking orchestrator at the Jetson

The accepted Jetson detector uses confidence threshold `0.69`. The DGX model
retains `0.70`. Keep this setting in an explicit Jetson configuration file.

Create the deployment-specific settings file:

```bash
cp \
  tracking-orchestrator/config/settings.yaml \
  tracking-orchestrator/config/settings.jetson.yaml
```

Edit `tracking-orchestrator/config/settings.jetson.yaml`:

```yaml
triton:
  url: "${TRITON_GRPC_URL}"
  detector_static_batch_size: 1
  depth_enabled: false

pipeline:
  detector_confidence: 0.69
```

Create a local Compose override named `docker-compose.jetson-clients.yml`:

```yaml
services:
  tracking-orchestrator:
    environment:
      TRITON_GRPC_URL: "${JETSON_TRITON_GRPC_URL:?Set the Jetson Triton gRPC endpoint}"
      ORCHESTRATOR_CONFIG_PATH: /work/continuous-tracking/tracking-orchestrator/config/settings.jetson.yaml
    volumes:
      - ./tracking-orchestrator/config/settings.jetson.yaml:/work/continuous-tracking/tracking-orchestrator/config/settings.jetson.yaml:ro
```

Start the orchestrator without starting the local DGX Triton dependency:

```bash
export JETSON_HOST=jetson-hostname-or-ip
export JETSON_TRITON_GRPC_URL="${JETSON_HOST}:8701"

docker compose stop triton

docker compose \
  -f docker-compose.yml \
  -f docker-compose.jetson-clients.yml \
  up -d --build --no-deps tracking-orchestrator
```

The Redis, database, MinIO, Cognitive Companion, and person-identification
dependencies must already be reachable.

Verify:

```bash
docker logs -f cts-orchestrator
curl -fsS http://localhost:8500/health
```

## 13. Monitor the Jetson

Watch Jetson telemetry:

```bash
tegrastats
```

Inspect Triton metrics:

```bash
curl -fsS http://localhost:8702/metrics \
  | grep -E 'nv_inference_(request|count|exec|queue|compute)'
```

Monitor:

- request, execution, and queue counts for all eight models;
- p95 detector latency and end-to-end frame latency;
- available memory and swap activity;
- GPU clocks, temperature, and thermal throttling;
- CTS stale-frame drops and inference queue growth.

Start production qualification with six cameras. Increase to eight only after
the latency, memory, drop-rate, and thermal gates pass under sustained motion.

## 14. Restore graphical boot

If the Jetson should normally boot into its desktop:

```bash
sudo systemctl set-default graphical.target
sudo systemctl start display-manager
```

Recheck Triton memory headroom after restoring the graphical session.

## 15. Preserve model artifacts

### Git LFS configuration

The repository tracks ONNX files through Git LFS:

```gitattributes
*.onnx filter=lfs diff=lfs merge=lfs -text
*.onnx_data filter=lfs diff=lfs merge=lfs -text
```

The generated Q/DQ models are intentionally ignored during calibration work.
After validation and license review, add them explicitly:

```bash
git lfs install

git add \
  .gitattributes \
  .gitignore \
  docker-compose.jetson-triton.yml \
  docs/jetson-int8-deployment.md \
  triton-models-jetson/README.md \
  triton-models-jetson/*/config.pbtxt \
  triton-models-jetson/scripts/

git add -f triton-models-jetson/*/1/model_int8.onnx

git lfs status
git status
```

Confirm that every ONNX file is represented by an LFS pointer:

```bash
git lfs ls-files | grep 'triton-models-jetson/.*/model_int8.onnx'
```

Commit and push only after reviewing the staged files:

```bash
git commit -m "Add validated Jetson INT8 model artifacts"
git push
```

GitHub LFS storage and bandwidth limits apply. Check the repository account's
current quota before pushing approximately 609 MB of Q/DQ models.

### Licensing before publication

Open-source application code does not automatically grant redistribution rights
for downloaded model weights or derived quantized weights.

In particular, InsightFace states that its provided pretrained models are for
non-commercial research and requests separate licensing contact for the
Buffalo_L package. Non-commercial household use does not by itself establish a
right to publish those weights in a public repository.

Before pushing any model or derived Q/DQ graph:

1. identify the exact source checkpoint;
2. preserve its license and attribution;
3. verify whether redistribution and derivative model publication are allowed;
4. obtain permission when the terms are unclear;
5. use private artifact storage until publication rights are confirmed.

Relevant upstream notices:

- [InsightFace license notice](https://github.com/deepinsight/insightface#license)
- [Ultralytics licensing](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- [MMPose license](https://github.com/open-mmlab/mmpose/blob/main/LICENSE)
- [SOLIDER repository](https://github.com/tinyvision/SOLIDER)

This runbook is operational guidance, not legal advice.

### Do not commit TensorRT plans

Keep these ignored:

```text
triton-models-jetson/**/1/model.plan
triton-models-jetson/reports/
calibration-data/
```

Plans are reproducible from the Q/DQ ONNX files and the recorded target stack.
If a plan backup is required for disaster recovery, store it privately with
the Jetson Linux, TensorRT, image tag, model checksum, and target board
metadata. Do not treat it as portable.

## Troubleshooting

### `docker stop` requires at least one argument

There are no running containers. Use the guarded form:

```bash
containers="$(docker ps -q)"
if [ -n "$containers" ]; then
  docker stop $containers
fi
```

### `nvidia-jetpack` is not installed

Inspect `nvidia-l4t-*`, CUDA, and TensorRT component packages. A missing
metapackage alone does not mean the Jetson driver stack is absent.

### Container requests desktop driver 560.28

Use only an `-igpu` Triton image intended for Jetson. Run the small
gender/age-engine test with `NVIDIA_DISABLE_REQUIRE=1`. Continue only if
TensorRT detects Orin and finishes with `PASSED`.

### TensorRT reports several gigabytes available but tactics see about 100 MB

Check `tegrastats`. A low `lfb` count indicates fragmented unified memory.
Compact memory or reboot into `multi-user.target`, then build before starting
other services.

### TensorRT reports no implementation for the first YOLO convolution

Read the preceding tactic warnings. If all tactics were skipped for insufficient
memory, fix memory fragmentation. Do not interpret the final optimizer error as
proof that the ONNX operator is unsupported.

### A model is not ready

Inspect:

```bash
docker logs cts-triton-jetson
curl -fsS http://localhost:8700/v2/repository/index
find triton-models-jetson -name model.plan -exec ls -lh {} \;
```

Confirm that the model name, `config.pbtxt`, tensor names, and plan file agree.

### Triton is killed while loading all models

Reduce unrelated services and graphical memory use. Do not reduce the number of
required Buffalo_L models because the person-identification service fails
closed when any required model is unavailable. If the complete set does not
fit with operational headroom, move the face service back to the DGX or use a
larger Jetson.

## References

- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [Triton model repository documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)
- [TensorRT `trtexec`](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html)
- [TensorRT explicit quantization](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [Triton metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html)
- [Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/)
- [Git LFS](https://git-lfs.com/)
