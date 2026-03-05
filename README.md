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
3. Export full-INT8 MobileNetV2 (include_top=True)
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
4. Verify it is really INT8
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="mobilenetv2_full_int8_top.tflite")
interpreter.allocate_tensors()

print("Input dtype:", interpreter.get_input_details()[0]["dtype"])
print("Output dtype:", interpreter.get_output_details()[0]["dtype"])
print("Input shape:", interpreter.get_input_details()[0]["shape"])

Expected output:

Input dtype: int8
Output dtype: int8
Input shape: [1 224 224 3]

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
