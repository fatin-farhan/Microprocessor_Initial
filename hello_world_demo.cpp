#include "fsl_debug_console.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_log.h"

// Implemented in hello_world_test.cc
TfLiteStatus ProfileMemoryAndLatency();
TfLiteStatus LoadFloatModelAndPerformInference();
TfLiteStatus LoadQuantModelAndPerformInference();

void RunHelloWorldTFLM()
{
  tflite::InitializeTarget();

  PRINTF("Running ProfileMemoryAndLatency...\r\n");
   if (ProfileMemoryAndLatency() != kTfLiteOk) {
     PRINTF("ProfileMemoryAndLatency FAILED\r\n");
     return;
  }

  PRINTF("Running LoadFloatModelAndPerformInference...\r\n");
  if (LoadFloatModelAndPerformInference() != kTfLiteOk) {
    PRINTF("LoadFloatModelAndPerformInference FAILED");
    return;
  }

  PRINTF("Running LoadQuantModelAndPerformInference...\r\n");
  if (LoadQuantModelAndPerformInference() != kTfLiteOk) {
    PRINTF("LoadQuantModelAndPerformInference FAILED");
    return;
  }

  PRINTF("~~~ALL TESTS PASSED~~~\r\n");
}