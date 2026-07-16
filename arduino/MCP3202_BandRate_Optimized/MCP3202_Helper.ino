// ====================================================================
// MCP3202 helper functions
// ====================================================================

inline void cs_low() {
  *cs_port_out &= ~cs_pin_mask;
}

inline void cs_high() {
  *cs_port_out |= cs_pin_mask;
}

inline uint16_t read_mcp3202_channel(uint8_t channel) {
  cs_low();

  SPI.transfer(0x01);
  uint8_t command = channel == 0 ? 0xA0 : 0xE0;
  uint8_t msb = SPI.transfer(command);
  uint8_t lsb = SPI.transfer(0x00);

  cs_high();

  return ((uint16_t)(msb & 0x0F) << 8) | lsb;
}

inline uint16_t read_mcp3202_ch0() {
  return read_mcp3202_channel(0);
}

inline uint16_t read_mcp3202_ch1() {
  return read_mcp3202_channel(1);
}

inline float raw_to_volts(uint16_t raw) {
  return raw * (MCP_REF_V / 4095.0f);
}

inline uint16_t read_one_raw_sample(int band_index) {
  return band_index == LOW_BAND_INDEX
      ? read_mcp3202_ch0()
      : read_mcp3202_ch1();
}
