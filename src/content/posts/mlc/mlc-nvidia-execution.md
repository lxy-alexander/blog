---
title: "mlc-nvidia-execution"
published: 2026-05-21
description: "mlc-nvidia-execution"
image: ""
tags: ["mlc","mlc-nvidia-execution"]
category: mlc
draft: false
lang: ""
createdAt: "2026-05-21T20:39:14.557.599780749Z"
---


# MLPerf Inference v6.0 — ResNet50 on H100-NVL-94GBx1

End-to-end guide for running the **ResNet50 Offline** benchmark from
`inference_results_v6.0` (NVIDIA closed division) on a single
**H100-NVL-94GB** GPU, using an Apptainer container.

---

## 1. Prerequisites

- A Linux host with a working NVIDIA driver and an H100-NVL-94GB visible to `nvidia-smi`.
- `apptainer` (with `--fakeroot` support).
- A writable host directory (`HOST_HOME`) with enough free space for:
  - ImageNet validation set (~6.3 GB tar, ~6.7 GB extracted)
  - MLPerf container sandbox (~30 GB+)
  - Preprocessed data, models, and build artifacts

<br>

## 2. Environment Setup & Download Source Code

### 2.1 Set environment variables

```bash
export HOST_HOME=/data/home/xli49
export MLPERF_SCRATCH_PATH=$HOST_HOME/mlperf_scratch
```

### 2.2 Create the required directories

```bash
mkdir -p $MLPERF_SCRATCH_PATH/data/imagenet
mkdir -p $MLPERF_SCRATCH_PATH/models
mkdir -p $MLPERF_SCRATCH_PATH/preprocessed_data
mkdir -p $MLPERF_SCRATCH_PATH/containers
```

### 2.3 Clone the MLPerf v6.0 repository 

```bash
cd $HOST_HOME
git clone https://github.com/mlcommons/inference_results_v6.0.git

export NVIDIA_HOME="$HOST_HOME/inference_results_v6.0/closed/NVIDIA"
cd $NVIDIA_HOME
mkdir -p 3rdparty
git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git 3rdparty/trtllm
git clone --depth 1 https://github.com/mlcommons/inference.git 3rdparty/mlc-inference
```

### 2.4 Expected directory layout

```text
YOUR_HOST_HOME
├── mlperf_scratch/
│   ├── containers/
│   ├── data/
│   │   └── imagenet/
│   ├── models/
│   └── preprocessed_data/
└── inference_results_v6.0/
    └── closed/
        └── NVIDIA/
           └── 3rdparty/
               ├── mlc-inference/
               └── trtllm/
```

---

<br>

## 3. Dataset Preparation

### 3.1 Download the ImageNet validation set

```bash
cd "$HOST_HOME"
wget -O "$HOST_HOME/ILSVRC2012_img_val.tar" \
  "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
```

### 3.2 Extract the dataset

```bash
tar -xvf "$HOST_HOME/ILSVRC2012_img_val.tar" \
  -C $MLPERF_SCRATCH_PATH/data/imagenet | \
awk 'NR % 1000 == 0 {print "extracted", NR, "files:", $0}'
```

### 3.3 Verify the image count

```bash
find $MLPERF_SCRATCH_PATH/data/imagenet -name "*.JPEG" | wc -l
```

Expected output:

```text
50000
```

---

<br>

## 4. Container Setup

### 4.1 Pull the MLPerf inference container

Pull the NVIDIA MLPerf Inference container
([NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/mlperf/containers/mlperf-inference/tags?version=tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_aarch64)):

```bash
apptainer pull --force \
  "$MLPERF_SCRATCH_PATH/containers/mlperf-inference-v6.sif" \
  docker://nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86
```

### 4.2 Convert the SIF image to a writable sandbox

```bash
cd "$MLPERF_SCRATCH_PATH/containers"

apptainer build --sandbox mlperf-inference-v6-sandbox mlperf-inference-v6.sif
```

Create mount points inside the sandbox:

```bash
export SANDBOX="$MLPERF_SCRATCH_PATH/containers/mlperf-inference-v6-sandbox"

mkdir -p "$SANDBOX$MLPERF_SCRATCH_PATH"
mkdir -p "$SANDBOX/work"
```

### 4.3 Enter the container

Select the target GPU index via `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES`
(check `nvidia-smi` first to confirm the index):

```bash
cd $NVIDIA_HOME

APPTAINERENV_HOME="$HOST_HOME" \
APPTAINERENV_CUDA_VISIBLE_DEVICES=0 \
APPTAINERENV_NVIDIA_VISIBLE_DEVICES=0 \
apptainer shell --nv --writable --fakeroot \
  --bind "$(pwd)":/work \
  --bind "$MLPERF_SCRATCH_PATH:$MLPERF_SCRATCH_PATH" \
  --env MLPERF_SCRATCH_PATH="$MLPERF_SCRATCH_PATH" \
  --pwd /work \
  "$SANDBOX"
```



---

## 5. Preprocessing 
### 5.1 Download data & model

```
make link_dirs 
make download_data BENCHMARKS="resnet50"
make download_model BENCHMARKS="resnet50"
```



### 5.2 Preprocess the dataset

Run the ResNet50 data preprocessing step:

```bash
cd /work
BENCHMARKS=resnet50 make preprocess_data
```

If the preprocessing step fails with the following OpenCV error:

`ImportError: libX11.so.6: cannot open shared object file`

the most likely cause is that both `opencv-python` and `opencv-python-headless` are installed in the same Python environment. Both packages provide the same Python module:`import cv2` and both write files under the same directory: `/usr/local/lib/python3.12/dist-packages/cv2/` . This can leave the environment in a mixed state: the package metadata for both packages remains, while the actual `cv2` files may have been partially overwritten. As a result, `import cv2` may load the OpenCV build with GUI dependencies, which then looks for `libX11.so.6`.

==To fix this, remove both OpenCV packages and reinstall only the headless version:==

```bash
pip uninstall -y opencv-python opencv-python-headless
pip install "opencv-python-headless==4.11.0.86"
```

Verify that OpenCV imports correctly:` python3 -c "import cv2; print('cv2', cv2.__version__, cv2.__file__)"` and its expected output: `cv2 4.11.0 /usr/local/lib/python3.12/dist-packages/cv2/__init__.py`

After `cv2` imports successfully, rerun the preprocessing step:

```bash
cd /work
BENCHMARKS=resnet50 make preprocess_data
```


Verify the preprocessed `.npy` files were generated:

```bash
find fp32         -type f -name "*.npy" | wc -l
find int8_chw4    -type f -name "*.npy" | wc -l
find int8_linear  -type f -name "*.npy" | wc -l
```



## 6. H100-NVL Configuration

### 6.1 Create the H100-NVL-94GBx1 Offline config

Create the config directory and file:

```bash
mkdir -p $NVIDIA_HOME/configs/H100-NVL-94GBx1/Offline
vim $NVIDIA_HOME/configs/H100-NVL-94GBx1/Offline/resnet50.py
```

Paste the following content into
**`closed/NVIDIA/configs/H100-NVL-94GBx1/Offline/resnet50.py`**:

```python
import code.common.constants as C
import code.fields.harness as harness_fields
import code.fields.loadgen as loadgen_fields
import code.fields.models as model_fields
from nvmitten.constants import Precision

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: {
            "resnet50": 256,
        },
        model_fields.precision: Precision.INT8,
        model_fields.input_dtype: Precision.INT8,
        model_fields.input_format: "linear",
        harness_fields.tensor_path: "build/preprocessed_data/imagenet/ResNet50/int8_linear",
        harness_fields.map_path: "data_maps/imagenet/val_map.txt",
        harness_fields.gpu_copy_streams: 4,
        harness_fields.gpu_inference_streams: 4,
        harness_fields.use_graphs: True,
        harness_fields.warmup_duration: 5.0,
        loadgen_fields.performance_sample_count: 1024,
        loadgen_fields.offline_expected_qps: 12500,
    },
}
```

### 6.2 Patch the accuracy checker

Edit **`closed/NVIDIA/code/common/mlcommons/accuracy_checker.py`** to register
a ResNet50 accuracy checker. Add the new class:

```diff
        env = dict()
        return _AccuracyScriptCommand(str(self.venv_path / "bin" / "python3"), argv, env)

+ @autoconfigure
+ class ResNet50AccuracyChecker(AccuracyChecker):
+     """Accuracy checker implementation for ResNet50 benchmark."""
+ 
+     def __init__(self, wl: Workload):
+         super().__init__(wl, "vision/classification_and_detection/tools/accuracy-imagenet.py")
+         self.val_map_path = paths.WORKING_DIR / "data_maps" / "imagenet" / "val_map.txt"
+ 
+     def get_cmd(self) -> _AccuracyScriptCommand:
+         argv = [paths.MLCOMMONS_INF_REPO / self.mlcommons_module_path,
+                 f"--mlperf-accuracy-file {self.log_file}",
+                 f"--imagenet-val-file {self.val_map_path}",
+                 "--dtype int32"]
+         return _AccuracyScriptCommand("python3", argv, dict())



G_ACCURACY_CHECKER_MAP = {C.Benchmark.BERT: BERTAccuracyChecker,
                          C.Benchmark.DLRMv2: DLRMv2AccuracyChecker,
@@ -939,7 +955,9 @@ def get_cmd(self) -> _AccuracyScriptCommand:
                          C.Benchmark.RGAT: RGATAccuracyChecker,
                          C.Benchmark.SDXL: SDXLAccuracyChecker,
                          C.Benchmark.WHISPER: WhisperAccuracyChecker,
-                           C.Benchmark.WAN22_A14B: Wan22AccuracyChecker}
+                           C.Benchmark.WAN22_A14B: Wan22AccuracyChecker,
+                           C.Benchmark.ResNet50: ResNet50AccuracyChecker
+                           }
"""Dict[Benchmark, AccuracyChecker]: Maps a Benchmark to its AccuracyChecker"""

```



### 6.3 Patch the plugin map

Edit **`closed/NVIDIA/code/plugin/__init__.py`** to add a ResNet50 entry to
`base_plugin_map` (ResNet50 does not require an external TensorRT plugin, so an
empty list is correct):

```diff
  base_plugin_map = {
+     Benchmark.ResNet50: [],
      Benchmark.DLRMv2: [LoadablePlugins.DLRMv2EmbeddingLookupPlugin],
      Benchmark.Retinanet: [LoadablePlugins.NMSOptPlugin, LoadablePlugins.RetinaNetConcatOutputPlugin],
  }
```

### 6.4 Disable the fusion

Edit **`closed/NVIDIA/code/resnet50/tensorrt/rn50_graphsurgeon.py`** 

```diff
import numpy as np
import onnx
import onnx_graphsurgeon as gs
+ import os

from nvmitten.constants import Precision
from nvmitten.nvidia.builder import ONNXNetwork


        self.disable_beta1_smallk = disable_beta1_smallk

    def fuse_ops(self):
+        if os.environ.get("RN50_DISABLE_FUSIONS", "0") == "1":
+            logging.info("RN50_DISABLE_FUSIONS=1, skipping ResNet50 plugin fusions")
+            return
+ 
        Res2Mega = self.fuse_res2_mega
        Beta1Smallk = self.fuse_beta1_conv

```

---

<br>

## 7. Build Dependencies

> **Note.** The third-party submodules (`3rdparty/trtllm`,`3rdparty/mlc-inference`) were already cloned in §2.3

### 7.1 Build LoadGen

Rebuild and install the MLPerf LoadGen Python wheel:

```bash
cd /work/3rdparty/mlc-inference/loadgen
python3 setup.py bdist_wheel
pip install dist/*.whl
```

Then build the LoadGen C++ library (`-fPIC` is required so the harness can link it): 

```bash
cd /work/3rdparty/mlc-inference/loadgen
rm -rf build
mkdir build
cd build

cmake .. \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_CXX_FLAGS="-fPIC" \
  -DCMAKE_C_FLAGS="-fPIC"

make -j
```

### 7.2 Build the harness

```bash
cd /work
rm -rf build/harness

PYTHONPATH=/work:$PYTHONPATH \
LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH \
LIBRARY_PATH=/usr/local/tensorrt/lib:$LIBRARY_PATH \
CPATH=/usr/local/tensorrt/include:$CPATH \
CPLUS_INCLUDE_PATH=/usr/local/tensorrt/include:$CPLUS_INCLUDE_PATH \
make build_harness FFI_UTILS_DIR=/work/build/ffi_utils
```

Verify the harness binary:

```bash
cd /work
ls -lh build/bin/harness_default
```

---

<br>

## 8. Run the Benchmark

### 8.1 Generate the TensorRT engine

```bash
cd /work

rm /build/engines/H100-NVL-94GBx1/Offline/resnet50/gpu-resnet50-int8-b256.cp990.plan

RN50_DISABLE_FUSIONS=1 SYSTEM_NAME=H100-NVL-94GBx1 \
make generate_engines RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --force"
```

### 8.2 Run the MLPerf harness (Performance)

```bash
cd /work

SYSTEM_NAME=H100-NVL-94GBx1 \
make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --test_mode=PerformanceOnly"
```

### 8.3 Review result logs

```bash
find build/logs -name "mlperf_log_summary.txt"
find build/logs -name "mlperf_log_detail.txt"
```

Typical output files:

```text
mlperf_log_summary.txt
mlperf_log_detail.txt
```

For `AccuracyOnly` mode, an accuracy log is also generated.

```bash
cd /work

SYSTEM_NAME=H100-NVL-94GBx1 \
make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --test_mode=AccuracyOnly"
```

---



<br>

<br>

# MLPerf Inference v6.0 — Llama3.1-8B on H100-NVL-94GBx1

## 1.Start the MLPerf Container

### 1.1 Set Host Paths and Enter the NVIDIA Repository

```bash
export HOST_HOME=/data/home/xli49
export MLPERF_SCRATCH_PATH=$HOST_HOME/mlperf_scratch
export NVIDIA_HOME="$HOST_HOME/inference_results_v6.0/closed/NVIDIA"

cd $NVIDIA_HOME
```

### 1.2 Start the Writable Sandbox Container

```bash
export SANDBOX="$MLPERF_SCRATCH_PATH/containers/mlperf-inference-v6-sandbox"

apptainer shell --nv --writable \
  --bind "$(pwd)":/work \
  --bind "$MLPERF_SCRATCH_PATH:$MLPERF_SCRATCH_PATH" \
  --env MLPERF_SCRATCH_PATH="$MLPERF_SCRATCH_PATH" \
  --pwd /work \
  "$SANDBOX"
```

This command mounts the current MLPerf repository to `/work` inside the container and keeps the scratch directory available at the same path.

the alternative is temporary writable overlay. This method uses a temporary writable filesystem. It is ==not recommended for== persistent changes because modifications are lost after the container exits.

```bash
apptainer shell --nv --writable-tmpfs \
  --bind "$(pwd)":/work \
  --bind "$MLPERF_SCRATCH_PATH:$MLPERF_SCRATCH_PATH" \
  "$MLPERF_SCRATCH_PATH/containers/mlperf-inference-v6.sif"
```

------

<br>

## 2. Download the Llama3.1-8B Model

Clone the Original Hugging Face Model Inside the container, use `/work` as the MLPerf repository root.

```bash
cd /work

export CHECKPOINT_PATH=build/models/Llama3.1-8B/Meta-Llama-3.1-8B-Instruct

git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct ${CHECKPOINT_PATH}

cd ${CHECKPOINT_PATH}
git checkout 0e9e39f249a16976918f6564b8830bc894c89659
cd /work
```

The model repository uses Git LFS, so `git-lfs` must be available inside the container.

------

<br>

## 3. Download Quantized Checkpoints

==H100 does not support FP4 execution for this workflow, so download both NVFP4 for the future and FP8 checkpoints==, but use FP8 for H100 runs.

```bash
cd /work

git clone https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-NVFP4 \
  build/models/Llama3.1-8B/fp4-quantized-modelopt

git clone https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8 \
  build/models/Llama3.1-8B/fp8-quantized-modelopt/llama3_1-8b-instruct-hf-torch-fp8
```

------

<br>

## 4. Download the Dataset

The required MLPerf benchmark data files are downloaded from the official MLCommons object storage.

-   `cnn_eval.json` is the official evaluation dataset. The MLPerf harness uses the articles in this file as inputs for Llama3.1-8B summarization. It is used for ==benchmark accuracy and performance evaluation==.
-   `cnn_dailymail_calibration.json` is the calibration dataset. It is used when generating the TensorRT-LLM engine or quantized model, so the runtime can observe representative input distributions.

```bash
cd /work

curl -L -o /tmp/mlc-r2-downloader.sh \
  https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh

bash /tmp/mlc-r2-downloader.sh \
  -d build/data/llama3.1-8b \
  https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri

bash /tmp/mlc-r2-downloader.sh \
  -d build/data/llama3.1-8b \
  https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-dailymail-calibration.uri
```

You can use the command below to check the structure of data:

```bash
python3 - <<'PY'
import json
from pathlib import Path

for p in [
    Path("build/data/llama3.1-8b/cnn_eval.json"),
    Path("build/data/llama3.1-8b/cnn_dailymail_calibration.json"),
]:
    data = json.load(open(p))
    x = data[0]
    print("\n==", p, "==")
    print("total:", len(data))
    print("fields:", list(x.keys()))
    print("\n--- instruction ---")
    print(x.get("instruction"))
    print("\n--- input ---")
    print(x.get("input"))
    print("\n--- output ---")
    print(x.get("output"))
    print("\n--- tok_input length ---")
    print(len(x.get("tok_input", [])))
    print("\n--- tok_input first 50 ---")
    print(x.get("tok_input", [])[:50])
PY
```

------

<br>

## 5. Preprocess the Dataset

Run Dataset Preprocessing

```bash
cd /work

python3 code/llama3_1-8b/tensorrt/preprocess_data.py \
  --data_dir build/data/ \
  --preprocessed_data_dir build/preprocessed-data
```

This step converts the raw CNN/DailyMail JSON files into the format expected by the MLPerf Llama3.1-8B benchmark pipeline.

------

<br>

## 6. Start the TensorRT-LLM Server

MPI Spawn is not supported properly inside this Apptainer setup. For a single-GPU run, the workaround is to bypass MPI Spawn and force TensorRT-LLM to use the local IPC executor.

### 6.1 Fix OpenMPI and Stop Old Processes

```bash
export OPAL_PREFIX=/opt/hpcx/ompi
export PATH=/opt/hpcx/ompi/bin:$PATH
export LD_LIBRARY_PATH=/opt/hpcx/ompi/lib:$LD_LIBRARY_PATH

pkill -9 -f "trtllm-serve|python3 -m code.main.*run_harness|make run_harness"
ps -ef | grep -E "trtllm|run_harness|30000" | grep -v grep
```

### 6.2 Patch TensorRT-LLM Executor for Single-GPU Execution

```bash
SITE=/usr/local/lib/python3.12/dist-packages/tensorrt_llm/executor/executor.py
cp $SITE ${SITE}.bak.single_gpu

python3 - <<'PY'
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/tensorrt_llm/executor/executor.py")
s = p.read_text()

needle = "        if not platform.system() == 'Windows':\n"

patch = """        if model_world_size == 1:
            return GenerationExecutor._create_ipc_executor(
                worker_kwargs,
                model_world_size=model_world_size,
                mpi_session=mpi_session,
                postproc_worker_config=postproc_worker_config,
                is_llm_executor=is_llm_executor,
                use_worker=True)

"""

if "if model_world_size == 1:" not in s:
    s = s.replace(needle, patch + needle, 1)
    p.write_text(s)

print("patched" if "if model_world_size == 1:" in p.read_text() else "patch failed")
PY
```

This patch makes TensorRT-LLM use the local IPC executor when `model_world_size == 1`, avoiding the MPI Spawn path.

### 6.3 Start the TensorRT-LLM Endpoint Server

```bash
cd /work

make run_llm_server RUN_ARGS="--core_type=trtllm_endpoint --benchmarks=llama3.1-8b --scenarios=Offline"
```

Use the command below to inspect the latest log file:

```
latest=$(find /work/build/logs -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
echo "$latest"

grep -niC 5 -E "error|exception|traceback|failed|fatal|abort|oom|cuda|cublas|nccl|mpi|spawn|connection refused" -- "$latest"
```

A healthy server startup should include messages similar to:

```tex
_config=None enable_sleep=False disable_flashinfer_sampling=False max_stats_len=1000
456-[05/21/2026-19:40:24] [TRT-LLM] [I] PROMETHEUS_MULTIPROC_DIR: /tmp/tmplkaz118m
457-INFO:     Started server process [73089]
458-INFO:     Waiting for application startup.
459-INFO:     Application startup complete.
```

This indicates that the TensorRT-LLM server is running successfully.

------

<br>

## 7. Run the Benchmark

```bash
cd /work

make run_harness RUN_ARGS="--core_type=trtllm_endpoint --benchmarks=llama3.1-8b --scenarios=Offline"
```

The harness connects to the TensorRT-LLM endpoint server and runs the Offline scenario for the Llama3.1-8B benchmark.



![image-20260521161033676](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260521161033676)

<br>
