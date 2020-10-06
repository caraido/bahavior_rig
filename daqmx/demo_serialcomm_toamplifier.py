# this demo is meant to set and get data on the nexus amplifier using the serial port
# in practice we can turn on the amplifier and adjust its settings when starting an experiment
# this will help us achieve data integrity
# we can also run tests before starting an experiment to verify that the amplifier is functioning properly

import serial

# this amplifier appears to use all the default settings, but define them explicitly here for clarity
settings = {
    'port': '/dev/name_of_amp',
    'baudrate': 9600,
    'bytesize': serial.EIGHTBITS,
    'parity': serial.PARITY_NONE,
    'stopbits': serial.STOPBITS_ONE,
}

with amp as serial.Serial(**settings):
  # message = amp.read(1) # reads a single byte
  # message = amp.readline() #read up until '\n'

  # amp.write(b'hello') #writes a string
  # see: https://pyserial.readthedocs.io/en/latest/pyserial_api.html

  # need to refer to amplifier manual to determine api
