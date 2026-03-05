## Step 0: File Structure Idea

```
tflite-micro/
├── tensorflow/
│   └── lite/
│       └── micro/
│           ├── models/
│           │   ├── mobilenetv2_int8.tflite
│           │   └── BUILD
│           │
│           └── examples/
│               └── mobilenetv2/
│                   ├── BUILD
│                   ├── main.cc
│                   ├── main_functions.cc
│                   ├── main_functions.h
│                   ├── model_settings.h
│                   ├── model_settings.cc
│                   ├── labels.h
│                   ├── labels.cc
│                   ├── detection_responder.h
│                   ├── detection_responder.cc
│                   ├── image_provider.h
│                   ├── image_provider.cc
│                   └── testdata/
│                       └── cat.bmp
```
## Step 1 Download a quantized MobileNetV2 .tflite
```bash
python3 -m venv tf
source tf/bin/activate
pip install --upgrade pip
pip install tensorflow pillow numpy
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

```bash
import tensorflow as tf

# Must be 224x224 if include_top=True with imagenet weights (Keras apps)
model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=True,
    weights="imagenet",
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Keep it simple first (float32 model)
tflite_model = converter.convert()

out_path = "mobilenet_v2_224_imagenet.tflite"
with open(out_path, "wb") as f:
    f.write(tflite_model)

print(f"Wrote {out_path}")

```
## Step 2 

``` colab
!pip -q install tensorflow tensorflow-datasets
import tensorflow as tf
import tensorflow_datasets as tfds

IMG_SIZE = 224
REP_SAMPLES = 200

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

dataset = tfds.load("tf_flowers", split="train", as_supervised=True)

calib_ds = (
    dataset.shuffle(2000)
           .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
           .batch(1)
           .take(REP_SAMPLES)
)

def representative_dataset():
    for x, _ in calib_ds:
        yield [x]
######
model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    weights="imagenet",
    include_top=True
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("mobilenetv2_full_int8_top.tflite","wb") as f:
    f.write(tflite_model)

print("Model exported")
#####
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="mobilenetv2_full_int8_top.tflite")
interpreter.allocate_tensors()

print("Input dtype:", interpreter.get_input_details()[0]["dtype"])
print("Output dtype:", interpreter.get_output_details()[0]["dtype"])
print("Input shape:", interpreter.get_input_details()[0]["shape"])

####

#Input dtype: int8
#Output dtype: int8
#Input shape: [1 224 224 3]

import tensorflow as tf
m = tf.lite.Interpreter(model_path="mobilenetv2_int8.tflite")
m.allocate_tensors()
print("INPUT:", m.get_input_details())
print("OUTPUT:", m.get_output_details())


from PIL import Image
# open image
img = Image.open("cat.jpeg")
# resize to 224x224
img_resized = img.resize((224, 224))
# save result
img_resized.save("cat.jpg")

```
Put it here:
```
tensorflow/lite/micro/models/mobilenetv2_int8.tflite
```
## Step 2 Generate model C arrays in micro/models/BUILD
Edit: tensorflow/lite/micro/models/BUILD
```
generate_cc_arrays(
    name = "generated_mobilenetv2_int8_model_cc",
    src = "mobilenetv2_int8.tflite",
    out = "mobilenetv2_int8_model_data.cc",
)

generate_cc_arrays(
    name = "generated_mobilenetv2_int8_model_hdr",
    src = "mobilenetv2_int8.tflite",
    out = "mobilenetv2_int8_model_data.h",
)
```

## Create the example BUILD (mobilenetv2)
Create/edit: tensorflow/lite/micro/examples/mobilenetv2/BUILD

```
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "generate_cc_arrays",
    "tflm_cc_binary",
    "tflm_cc_library",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

# Embed cat.bmp into a byte array (this embeds bytes; decoding/quantization is in code)
generate_cc_arrays(
    name = "generated_cat_cc",
    src = "testdata/cat.bmp",
    out = "cat_image_data.cc",
)

generate_cc_arrays(
    name = "generated_cat_hdr",
    src = "testdata/cat.bmp",
    out = "cat_image_data.h",
)

tflm_cc_library(
    name = "cat_image_data",
    srcs = [":generated_cat_cc"],
    hdrs = [":generated_cat_hdr"],
)

tflm_cc_library(
    name = "model_settings",
    srcs = ["model_settings.cc"],
    hdrs = ["model_settings.h"],
)

tflm_cc_library(
    name = "labels",
    srcs = ["labels.cc"],
    hdrs = ["labels.h"],
    deps = [":model_settings"],
)

tflm_cc_library(
    name = "image_provider",
    srcs = ["image_provider.cc"],
    hdrs = ["image_provider.h"],
    deps = [
        ":cat_image_data",
        ":model_settings",
        "//tensorflow/lite/c:common",
    ],
)

tflm_cc_library(
    name = "detection_responder",
    srcs = ["detection_responder.cc"],
    hdrs = ["detection_responder.h"],
    deps = [
        ":labels",
        ":model_settings",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_log",
    ],
)

tflm_cc_library(
    name = "main_functions",
    srcs = ["main_functions.cc"],
    hdrs = ["main_functions.h"],
    deps = [
        ":detection_responder",
        ":image_provider",
        ":model_settings",
        "//tensorflow/lite/micro/models:mobilenetv2_int8_model_data",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_interpreter",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_mutable_op_resolver",
        "//tensorflow/lite/micro:system_setup",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

tflm_cc_binary(
    name = "mobilenetv2",
    srcs = ["main.cc"],
    deps = [
        ":main_functions",
        "//tensorflow/lite/micro:micro_log",
    ],
)

```

Step 4 model_settings.h
```
tensorflow/lite/micro/examples/mobilenetv2/model_settings.h
#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MOBILENETV2_MODEL_SETTINGS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MOBILENETV2_MODEL_SETTINGS_H_

constexpr int kNumCols = 224;
constexpr int kNumRows = 224;
constexpr int kNumChannels = 3;

constexpr int kCategoryCount = 1000;

#endif
model_settings.cc can be empty.

```
Step 5) Labels:
```
labels.h
#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MOBILENETV2_LABELS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MOBILENETV2_LABELS_H_

#include "tensorflow/lite/micro/examples/mobilenetv2/model_settings.h"

extern const char* kCategoryLabels[kCategoryCount];

#endif
labels.cc:

#include "tensorflow/lite/micro/examples/mobilenetv2/labels.h"

const char* kCategoryLabels[kCategoryCount] = {
  // 1000 strings in order
};

```

## 6) Image provider: embed BMP bytes + quantize to int8
```
image_provider.h:

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MOBILENETV2_IMAGE_PROVIDER_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MOBILENETV2_IMAGE_PROVIDER_H_

#include "tensorflow/lite/c/common.h"

TfLiteStatus GetImage(TfLiteTensor* input);

#endif

image_provider.cc
#include "tensorflow/lite/micro/examples/mobilenetv2/image_provider.h"
#include "tensorflow/lite/micro/examples/mobilenetv2/cat_image_data.h"

#include <cstdint>

static inline int8_t QuantizeU8ToI8(uint8_t u8, float scale, int zp) {
  float real = (float)u8 / 255.0f;
  int q = (int)(real / scale + (float)zp + 0.5f);
  if (q > 127) q = 127;
  if (q < -128) q = -128;
  return (int8_t)q;
}

TfLiteStatus GetImage(TfLiteTensor* input) {
  if (!input || input->type != kTfLiteInt8) return kTfLiteError;

  const int needed = input->bytes;
  if ((unsigned)needed != g_cat_image_data_size) return kTfLiteError;

  const float scale = input->params.scale;
  const int zp = input->params.zero_point;

  const uint8_t* src = g_cat_image_data;
  int8_t* dst = input->data.int8;

  for (int i = 0; i < needed; ++i) {
    dst[i] = QuantizeU8ToI8(src[i], scale, zp);
  }
  return kTfLiteOk;
}

```
## Step 7) main_functions.cc: load model, resolver ops, invoke
```
#include "tensorflow/lite/micro/examples/mobilenetv2/main_functions.h"

#include "tensorflow/lite/micro/examples/mobilenetv2/detection_responder.h"
#include "tensorflow/lite/micro/examples/mobilenetv2/image_provider.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/models/mobilenetv2_int8_model_data.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
bool g_ready = false;

constexpr int kTensorArenaSize = 2 * 1024 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void setup() {
  tflite::InitializeTarget();
  g_ready = false;

  model = tflite::GetModel(g_mobilenetv2_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Bad model schema");
    return;
  }

  // You may need to adjust this op list to your model.
  static tflite::MicroMutableOpResolver<12> r;
  r.AddConv2D(tflite::Register_CONV_2D_INT8());
  r.AddDepthwiseConv2D(tflite::Register_DEPTHWISE_CONV_2D_INT8());
  r.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D_INT8());
  r.AddReshape();
  r.AddSoftmax(tflite::Register_SOFTMAX_INT8());
  r.AddAdd();
  r.AddMul();
  r.AddRelu();
  r.AddPad();
  r.AddMaxPool2D();
  r.AddFullyConnected();
  r.AddMean();   // important if your model uses MEAN

  static tflite::MicroInterpreter static_interpreter(model, r, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors failed");
    return;
  }

  input = interpreter->input(0);
  MicroPrintf("input type=%d scale=%f zp=%d", input->type, input->params.scale, input->params.zero_point);
  g_ready = true;
}

void loop() {
  if (!g_ready) return;

  if (GetImage(input) != kTfLiteOk) {
    MicroPrintf("Image capture failed");
    return;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  TfLiteTensor* output = interpreter->output(0);
  RespondToDetection(output);
}

```

## Step 8) detection_responder.cc: print top1 label
```

#include "tensorflow/lite/micro/examples/mobilenetv2/detection_responder.h"
#include "tensorflow/lite/micro/examples/mobilenetv2/labels.h"
#include "tensorflow/lite/micro/micro_log.h"

void RespondToDetection(const TfLiteTensor* output) {
  const int8_t* s = output->data.int8;

  int best_i = 0;
  int8_t best_q = s[0];
  for (int i = 1; i < 1000; ++i) {
    if (s[i] > best_q) { best_q = s[i]; best_i = i; }
  }

  float score = ( (int)best_q - output->params.zero_point ) * output->params.scale;
  MicroPrintf("top1: %s (class=%d) score=%f q=%d",
              kCategoryLabels[best_i], best_i, score, (int)best_q);
}


```

## Step 9) Build + run

bazel clean
bazel build //tensorflow/lite/micro/examples/mobilenetv2:mobilenetv2
bazel run //tensorflow/lite/micro/examples/mobilenetv2:mobilenetv2

## Step 10 Now Write a Makefile.inc
```
# MobileNetV2 example (person_detection style)

mobilenetv2_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobilenetv2/main.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobilenetv2/main_functions.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobilenetv2/model_settings.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobilenetv2/image_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobilenetv2/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobilenetv2/labels.cc

mobilenetv2_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobilenetv2/main_functions.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobilenetv2/model_settings.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobilenetv2/image_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobilenetv2/detection_responder.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobilenetv2/labels.h

# Extra files that must be copied into the tflm-tree (not compiled)
mobilenetv2_DATA := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobilenetv2/testdata/cat.bmp

mobilenetv2_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/mobilenetv2_int8.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobilenetv2/testdata/cat.bmp

mobilenetv2_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/mobilenetv2_int8_model_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobilenetv2/testdata/cat_image_data.cc

mobilenetv2_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/mobilenetv2_int8_model_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobilenetv2/testdata/cat_image_data.h

# Build the example binary
$(eval $(call microlite_test,mobilenetv2,\
$(mobilenetv2_SRCS),$(mobilenetv2_HDRS),$(mobilenetv2_GENERATOR_INPUTS)))

# Add generated sources/headers (same pattern as person_detection)
mobilenetv2_SRCS += $(mobilenetv2_GENERATED_SRCS)
mobilenetv2_HDRS += $(mobilenetv2_GENERATED_HDRS)

# Required for create_tflm_tree.py
list_mobilenetv2_example_sources:
	@echo $(mobilenetv2_SRCS)

list_mobilenetv2_example_headers:
	@echo $(mobilenetv2_HDRS)
```
```
python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py -e mobilenetv2 --makefile_options="TARGET=cortex_m_generic TARGET_ARCH=cortex-m7" /home/ai/Desktop/tflm-tree-mobilenetv2
```

# Microprocessor_Initial
Setting up NXP MIMXRT1020EVK for DNN

# Installing Bazelisk
    curl -Lo bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
    chmod +x bazel
    sudo mv bazel /usr/local/bin/bazel

# Clone the Repository
    git clone https://github.com/tensorflow/tflite-micro.git
    cd tflite-micro

# Build the example
    bazel build tensorflow/lite/micro/examples/hello_world:hello_world_test
 
# Run the test
    bazel run tensorflow/lite/micro/examples/hello_world:hello_world_test
# Or
    make -f tensorflow/lite/micro/tools/make/Makefile test_hello_world_test

# Run the Person Detection Test
# Download required third-party dependencies
    make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads
 
# Build and run the test
    make -f tensorflow/lite/micro/tools/make/Makefile test_person_detection_test

# Bringing Up on a New Platform
    python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py -e hello_world -e person_detection /tmp/tflm-tree
    python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py -e hello_world --makefile_options="TARGET=cortex_m_generic TARGET_ARCH=cortex-m7" /tmp/tflm-tree-hello-world

## Add *tflm-tree-hello-world* to Project Directory.
## Include paths
Project->Properties->C/C++ General->Paths and Symbols ->Includes
- /${ProjName}/tflm-tree-hello-world/
- /${ProjName}/tflm-tree-hello-world/tensorflow
- /${ProjName}/tflm-tree-hello-world/tensorflow/lite
- /${ProjName}/tflm-tree-hello-world/tensorflow/lite/micro
- /${ProjName}/tflm-tree-hello-world/third_party/flatbuffers/include
- /${ProjName}/tflm-tree-hello-world/third_party/gemmlowp
- /${ProjName}/tflm-tree-hello-world/third_party/kissfft
- /${ProjName}/tflm-tree-hello-world/third_party/ruy
- /${ProjName}/tflm-tree-hello-world/examples/hello_world

## Add Preprocessor defines
Project->Properties->C/C++ General->Paths and Symbols ->Symbols-> GNU C++
- TF_LITE_STATIC_MEMORY

## Set C++ Standard
Project->Properties->C/C++ Build->Settings->Tool Settings->MCU C++ Compiler ->Dialect
- Set language standard to ISO C++17 or,
- Miscellaneous ->-std=c++17
## Linker flag for float print
Project Properties → C/C++ Build → Settings → MCU C Compiler → Preprocessor
Project Properties → C/C++ Build → Settings → MCU C++ Compiler → Preprocessor
PRINTF_FLOAT_ENABLE=1
PRINTF_ADVANCED_ENABLE=1
-u_print_float
## Including and excluding

Check if files are excluded from build or not. If excluded, include them.
Exclide tflm-cortex-m7/tensorflow/lite/micro/cortex_m_generic/micro_time.cc and add a dummy micro_time.cc inside source/
# Write source/tflm_hello_world.cc and source/MIMXRT1020_Project_Test.cpp
Write a program.cc and a header program.h
Call the program from main.cpp
