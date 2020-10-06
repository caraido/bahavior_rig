# this demo collects audio data from ai1 while sending triggers with ctr0 back to ai0
# data is saved to disk using automatic logging
# the most recent data is stored in memory and plotted at the end of the demo

import plotly.io as pio  # only for demo purposes
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, TaskMode
# from nidaqmx.stream_readers import AnalogSingleChannelReader as AnalogReader
from nidaqmx.stream_readers import AnalogMultiChannelReader as AnalogReader

# Constants
SAMPLE_RATE = int(3e5)  # audio sampling rate
TRIGGER_FREQ = 30  # video frame rate
DUTY_CYCLE = 0.01  # fraction of the time that video trigger is high... tbd
READ_RATE = 10  # NI recommends at least 10 Hz
BUFFER_DEPTH = 10  # how many read cycles to keep in memory
demo_time = 10  # time in seconds for this demo


with nidaqmx.Task() as audio, nidaqmx.Task() as video:
  # note that the with statement causes the tasks to be cleaned up upon exit
  # need one task per channel type

  # configure the audio triggering task
  audio.ai_channels.add_ai_voltage_chan(
      "Dev1/ai0"
  )  # this channel measures the counter signal (for demo purposes only)
  audio.ai_channels.add_ai_voltage_chan(
      "Dev1/ai1"
  )  # this channel measures the audio signal
  audio.timing.cfg_samp_clk_timing(
      SAMPLE_RATE, sample_mode=AcquisitionType.CONTINUOUS
  )  # acquire data at 300KHz until task is stopped
  audio.control(
      TaskMode.TASK_COMMIT
  )  # transition the task to the committed state so it's ready to start
  audio.in_stream.configure_logging('testing.tdms')  # see nptdms

  # configure the video triggering task
  video.co_channels.add_co_pulse_chan_freq(
      "Dev1/ctr0", freq=TRIGGER_FREQ, duty_cycle=DUTY_CYCLE
  )  # fires a TTL pulse at 30Hz lasting ~333 uSec
  video.triggers.start_trigger.cfg_dig_edge_start_trig(
      "/Dev1/ai/StartTrigger"
  )  # start the video trigger with the audio channel
  video.timing.cfg_implicit_timing(
      sample_mode=AcquisitionType.CONTINUOUS
  )  # configure the trigger to repeat until the task is stopped
  video.control(
      TaskMode.TASK_COMMIT
  )

  # configure reading
  read_count = SAMPLE_RATE//READ_RATE
  reader = AnalogReader(audio.in_stream)
  data = [np.ndarray(shape=(2, read_count)) for i in range(BUFFER_DEPTH)]

  video.start()
  audio.start()  # fires the ai start trigger, which should also start the video trigger sequence

  # begin data collection
  # the idea here is that we only want to keep the most recent samples in memory, so we overwrite
  # we can refactor so that other threads have access to the values in data
  for i in range(READ_RATE*demo_time):
    reader.read_many_sample(
        data[i % BUFFER_DEPTH],
        number_of_samples_per_channel=read_count
    )
    # we can do other tasks in here so long as we keep up with collection

  # demo: display 1 second of data
  data = np.hstack(data)
  fig = {
      "data": [{"type": "scatter", "y": data[1]}, {"type": "scatter", "y": data[0]}]
  }

  pio.show(fig)

# task will be cleaned up at end of 'with' context
