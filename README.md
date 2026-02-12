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
    python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py -e hello_world --makefile_options="TARGET=cortex_m_generic TARGET_ARCH=cortex-m7" /tmp/tflm-cortex-m7

## Add *tflm-cortex-m7* to Project Directory.
## Include paths
Project->Properties->C/C++ General->Paths and Symbols ->Includes
- tflm-cortex-m7
- tflm-cortex-m7/tensorflow
- tflm-cortex-m7/tensorflow/lite
- tflm-cortex-m7/tensorflow/lite/micro
- tflm-cortex-m7/third_party/flatbuffers/include
- tflm-cortex-m7/third_party/flatbuffers/gemmlowp
- tflm-cortex-m7/third_party/kissfft
- tflm-cortex-m7/third_party/ruy

## Add Preprocessor defines
Project->Properties->C/C++ General->Paths and Symbols ->Symbols-> GNU C++
- TF_LITE_STATIC_MEMORY

## Set C++ Standard
Project->Properties->C/C++ Build->Settings->Tool Settings->MCU C++ Compiler ->Dialect
- Set language standard to ISO C++17 or,
- Miscellaneous ->-std=c++17
## Linker flag for float print
-u_print_float
## Including and excluding

Check if files are excluded from build or not. If excluded, include them.
Exclide tflm-cortex-m7/tensorflow/lite/micro/cortex_m_generic/micro_time.cc and add a dummy micro_time.cc inside source/
# Write source/tflm_hello_world.cc and source/MIMXRT1020_Project_Test.cpp
Write a program.cc and a header program.h
Call the program from main.cpp