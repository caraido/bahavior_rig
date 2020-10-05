import nidaqmx
from nidaqmx.constants import AcquisitionType, TaskMode

import plotly.io as pio  # only for demo purposes
# here we assume that the output from ctr0 is connected to ai0 (using BNC)
# in practice we want ctr0 tied to the hardware trigger of all the cameras

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
      3e5, sample_mode=AcquisitionType.CONTINUOUS
  )  # acquire data at 300KHz until task is stopped
  audio.control(
      TaskMode.TASK_COMMIT
  )  # transition the task to the committed state so it's ready to start

  # configure the video triggering task
  video.co_channels.add_co_pulse_chan_freq(
      "Dev1/ctr0", freq=30, duty_cycle=0.01
  )  # fires a TTL pulse at 30Hz lasting ~333 uSec
  video.triggers.start_trigger.cfg_dig_edge_start_trig(
      "/Dev1/ai/StartTrigger"
  )  # start the video trigger with the audio channel
  video.control(
      TaskMode.TASK_COMMIT
  )

  audio.start()  # fires the ai start trigger, which should also start the video trigger sequence

  # demo: display 1 second of data
  data = audio.read(
      number_of_samples_per_channel=300000
  )

  fig = {
      "data": [{"type": "scatter", "y": data[0]}, {"type": "scatter", "y": data[1]}]
  }

  pio.show(fig)
