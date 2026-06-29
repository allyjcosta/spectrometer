# serial_io.py

import time
import serial
import numpy as np


def open_serial(port, baud, startup_delay_s=2.0):
    ser = serial.Serial(port, baud, timeout=0.05)
    time.sleep(startup_delay_s)
    print("Connected to:", ser.name)
    return ser


def send_cmd(ser, cmd):
    ser.write((cmd + "\n").encode())
    ser.flush()


def request_raw_sample_rate(ser):
    send_cmd(ser, "RATE")

    while True:
        line = ser.readline().decode(errors="ignore").strip()

        if line.startswith("RAW_SAMPLE_RATE_HZ:"):
            rate = float(line.split(":")[1])
            print(f"RAW_SAMPLE_RATE_HZ = {rate:.3f}")
            return rate


def read_one_band_block(ser):
    data = []

    while True:
        line = ser.readline().decode(errors="ignore").strip()

        if line.startswith("BAND:"):
            band_name = line.split(":", 1)[1]
            break

    while True:
        line = ser.readline().decode(errors="ignore").strip()

        if line == "START":
            break

    while True:
        line = ser.readline().decode(errors="ignore").strip()

        if line == "END":
            break

        try:
            data.append(float(line))
        except ValueError:
            pass

    return band_name, np.array(data)


def read_band_blocks(ser, band_order):
    blocks = {}

    send_cmd(ser, "BLOCK")

    while len(blocks) < len(band_order):
        band_name, data = read_one_band_block(ser)

        if band_name in band_order:
            blocks[band_name] = data
        else:
            print("Ignoring unknown band:", band_name)

    return blocks
