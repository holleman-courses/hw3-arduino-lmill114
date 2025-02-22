#include <Arduino.h>
#include <TensorFlowLite.h>
#include "model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/version.h"

#define INPUT_BUFFER_SIZE 64
#define OUTPUT_BUFFER_SIZE 64
#define INT_ARRAY_SIZE 7  // Expect exactly 7 inputs for the sine wave model

// Buffer for model execution
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// TFLM variables
const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
TfLiteTensor* input_tensor;
TfLiteTensor* output_tensor;

char in_str_buff[INPUT_BUFFER_SIZE];  // Input buffer
int input_array[INT_ARRAY_SIZE];      // Array for parsed integers
int in_buff_idx = 0;

void setup() {
  Serial.begin(9600);
  delay(5000);
  Serial.println("TFLM Sine Wave Prediction Model Initializing...");

  // Load the model
  model = tflite::GetModel(model_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    return;
  }

  // Initialize interpreter
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Failed to allocate tensors!");
    return;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.println("Model loaded successfully!");
}

int string_to_array(char *in_str, int *int_array) {
  int num_integers = 0;
  char *token = strtok(in_str, ",");
  while (token != NULL) {
    int_array[num_integers++] = atoi(token);
    token = strtok(NULL, ",");
    if (num_integers >= INT_ARRAY_SIZE) break;
  }
  return num_integers;
}

void measure_and_run_model() {
  unsigned long t0 = micros();
  Serial.println("Running inference...");
  unsigned long t1 = micros();

  // Check tensor type
  if (input_tensor->type != kTfLiteInt8 || output_tensor->type != kTfLiteInt8) {
    Serial.println("Error: Expected int8 tensors!");
    return;
  }

  // Get quantization parameters
  float input_scale = input_tensor->params.scale;
  int input_zero_point = input_tensor->params.zero_point;
  float output_scale = output_tensor->params.scale;
  int output_zero_point = output_tensor->params.zero_point;

  // Convert input values to int8 using quantization and ensure it fits the int8 range (-128 to 127)
  for (int i = 0; i < INT_ARRAY_SIZE; i++) {
    // Apply quantization to scale input within range [-128, 127]
    int8_t quantized_value = static_cast<int8_t>((input_array[i] - input_zero_point) / input_scale);
    input_tensor->data.int8[i] = quantized_value;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Model inference failed!");
    return;
  }

  unsigned long t2 = micros();

  // Get the raw int8 prediction
  int8_t raw_prediction = output_tensor->data.int8[0];

  // Convert back to floating-point (dequantized) and scale output to fit in range [-128, 127]
  float predicted_float = (raw_prediction - output_zero_point) * output_scale;

  // Print raw dequantized value before rounding
  Serial.print("Raw dequantized prediction (float): ");
  Serial.println(predicted_float);

  // Ensure that the predicted value is within the int8 range
  int predicted_integer = round(predicted_float);
  if (predicted_integer < -128) predicted_integer = -128;
  if (predicted_integer > 127) predicted_integer = 127;

  Serial.print("Predicted next integer value: ");
  Serial.println(predicted_integer);  // Print as integer

  Serial.print("Printing time (us): ");
  Serial.println(t1 - t0);
  Serial.print("Inference time (us): ");
  Serial.println(t2 - t1);
}

void process_input() {
  int num_values = string_to_array(in_str_buff, input_array);
  if (num_values != INT_ARRAY_SIZE) {
    Serial.println("Error: Please enter exactly 7 integers.");
    return;
  }

  measure_and_run_model();
}

void loop() {
  if (Serial.available()) {
    char received_char = Serial.read();
    Serial.print(received_char); // Echo
    in_str_buff[in_buff_idx++] = received_char;
    if (received_char == 13) { // Enter key
      Serial.println();
      process_input();
      memset(in_str_buff, 0, INPUT_BUFFER_SIZE);
      in_buff_idx = 0;
    } else if (in_buff_idx >= INPUT_BUFFER_SIZE) {
      memset(in_str_buff, 0, INPUT_BUFFER_SIZE);
      in_buff_idx = 0;
    }
  }
}
