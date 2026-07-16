// =====================================================
// Setup instrument
// =====================================================
void setup_instrument() {
  SPI.begin();
  pinMode(chipSelectPin, OUTPUT);
  digitalWrite(chipSelectPin, HIGH);

  cs_port_out = portOutputRegister(g_APinDescription[chipSelectPin].ulPort);
  cs_pin_mask = 1ul << g_APinDescription[chipSelectPin].ulPin;
  cs_high();
}

// =====================================================
// Calibration
// =====================================================
void calibrate_sample_rate() {
  SPI.beginTransaction(SPISettings(SPI_CLOCK_HZ, MSBFIRST, SPI_MODE0));

  unsigned long t0 = micros();
  for (int i = 0; i < SAMPLES; i++) {
    read_mcp3202_ch1();
  }
  unsigned long t1 = micros();

  SPI.endTransaction();

  raw_sample_rate_Hz = (SAMPLES * 1000000.0f) / (float)(t1 - t0);
}

// =====================================================
// Serial commands
// =====================================================
void check_serial_command() {
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();

  if (cmd == "RATE") {
    Serial.print("RAW_SAMPLE_RATE_HZ:");
    Serial.println(raw_sample_rate_Hz, 3);
  }
  else if (cmd == "BLOCK") {
    run_block_mode();
  }
  else if (cmd == "RAW") {
    print_raw_debug();
  }
}

void print_raw_debug() {
  SPI.beginTransaction(SPISettings(SPI_CLOCK_HZ, MSBFIRST, SPI_MODE0));

  uint16_t ch0 = read_mcp3202_ch0();
  uint16_t ch1 = read_mcp3202_ch1();

  SPI.endTransaction();

  Serial.print("MCP3202_CH0_RAW:");
  Serial.print(ch0);
  Serial.print(",V:");
  Serial.println(raw_to_volts(ch0), 9);

  Serial.print("MCP3202_CH1_RAW:");
  Serial.print(ch1);
  Serial.print(",V:");
  Serial.println(raw_to_volts(ch1), 9);
}

// =====================================================
// Block mode
// =====================================================
void run_block_mode() {
  for (int band_index = 0; band_index < NUM_BANDS; band_index++) {
    Serial.print("BAND:");
    Serial.println(band_names[band_index]);

    func_data_acq(band_index);

    Serial.print("BAND_SAMPLE_RATE_HZ:");
    Serial.println(band_sample_rate_Hz[band_index], 6);

    send_block();
  }
}

// =====================================================
// Read one averaged ADC sample
// =====================================================
uint16_t avg_read_sample(float f_min) {
  float sample_interval_us = 1000000.0f / (f_min * SAMPLES);
  unsigned long t_start = micros();

  uint32_t sum = 0;
  uint32_t count = 0;

  while ((micros() - t_start) < sample_interval_us) {
    sum += read_mcp3202_ch0();
    count++;
  }

  if (count == 0) count = 1;

  return (uint16_t)(sum / count);
}

// =====================================================
// Acquire full block
// =====================================================
void func_data_acq(int band_index) {
  unsigned long t0 = micros();

  SPI.beginTransaction(SPISettings(SPI_CLOCK_HZ, MSBFIRST, SPI_MODE0));

  if (band_index == HIGH_BAND_INDEX) {
    for (int i = 0; i < SAMPLES; i++) {
      sample_arr_V[i] = raw_to_volts(read_mcp3202_ch1());
    }
  }
  else {
    float f_min = f_min_list[band_index];
    for (int i = 0; i < SAMPLES; i++) {
      sample_arr_V[i] = raw_to_volts(avg_read_sample(f_min));
    }
  }

  SPI.endTransaction();

  unsigned long elapsed_us = micros() - t0;
  band_sample_rate_Hz[band_index] =
      (SAMPLES * 1000000.0f) / (float)elapsed_us;

  if (band_index == HIGH_BAND_INDEX) {
    raw_sample_rate_Hz = band_sample_rate_Hz[band_index];
    f_min_list[HIGH_BAND_INDEX] = raw_sample_rate_Hz / SAMPLES;
  }
}

// =====================================================
// Send full voltage block
// =====================================================
void send_block() {
  Serial.println("START");

  for (int i = 0; i < SAMPLES; i++) {
    Serial.println(sample_arr_V[i], 9);
  }

  Serial.println("END");
}
