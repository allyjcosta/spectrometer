#include <SPI.h>

// =====================================================
// MCP3202-only band-rate acquisition sketch
// =====================================================
//
// Serial protocol used by the Python spectrometer:
//   RATE  -> RAW_SAMPLE_RATE_HZ:<rate>
//   BLOCK -> LOW block, then HIGH block
//   RAW   -> quick raw ADU debug for CH0 and CH1
//
// Band inputs:
//   LOW  band -> MCP3202 CH0
//   HIGH band -> MCP3202 CH1
//
// Hardware:
//   MCP3202 CS -> SAMD21 A3
//   MCP3202 SCK/MOSI/MISO -> board SPI pins
//   MCP3202 Vref should match MCP_REF_V below

#define SAMPLES 4096
#define NUM_BANDS 2

#define LOW_BAND_INDEX  0
#define HIGH_BAND_INDEX 1

const int chipSelectPin = A3;

const float MCP_REF_V = 3.3f;
const uint32_t SPI_CLOCK_HZ = 1000000;

const char* band_names[NUM_BANDS] = {"LOW", "HIGH"};
float f_min_list[NUM_BANDS] = {0.1f, 0.0f};

float sample_arr_V[SAMPLES];
float raw_sample_rate_Hz = 0.0f;
float band_sample_rate_Hz[NUM_BANDS] = {0.0f, 0.0f};

volatile uint32_t* cs_port_out = nullptr;
uint32_t cs_pin_mask = 0;

// functions
void setup_instrument();
void calibrate_sample_rate();
void check_serial_command();
void run_block_mode();
void print_raw_debug();

void setup() {
  Serial.begin(2000000);
  while (!Serial) {}
  delay(1000);

  setup_instrument();
  calibrate_sample_rate();
  f_min_list[HIGH_BAND_INDEX] = raw_sample_rate_Hz / SAMPLES;
  check_serial_command();
}

void loop() {
  check_serial_command();
}
