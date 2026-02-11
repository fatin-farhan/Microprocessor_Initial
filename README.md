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
