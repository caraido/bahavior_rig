from AcquisitionGroup import AcquisitionGroup
from RigStatus import RigStatus
# from utils.audio_settings import audio_settings
from initialStatus import initialStatus


def setup():
  # global printToGUI  # TODO: should be defined in socketApp.py??
  # will look something like: socketio.emit("message", f"message content string")

  # global annotationsToGUI  # TODO: should be defined in socketApp.py??
  # will look something like socketio.emit("annotation", {"streamId":0, "data": [ {"rectangle":[(p0x, p0y), ..., (p3x, p3y)]}, ... ]})

  status = RigStatus(initialStatus)
  ag = AcquisitionGroup(status)

  # TODO: do this better
  for i in range(ag.nCameras):
    for j in range(ag.nCameras):
      if status[f'camera {i}'].current['serial number'].current == int(ag.cameras[j].device_serial_number):
        # thisCamera = status[f'camera {i}'].current
        break
      if j == ag.nCameras-1:
        raise Exception('Could not match cameras')

    status[f'camera {j}'].current['width'].mutable()
    status[f'camera {j}'].current['height'].mutable()
    status[f'camera {j}'].current['port'].mutable()

    status[f'camera {j}'].current['width'](ag.cameras[i].width)
    status[f'camera {j}'].current['height'](ag.cameras[i].height)
    status[f'camera {j}'].current['port'](ag.cameras[i].address[1])

    status[f'camera {j}'].current['width'].immutable()
    status[f'camera {j}'].current['height'].immutable()
    status[f'camera {j}'].current['port'].immutable()

  status['spectrogram'].current['port'].mutable()
  status['spectrogram'].current['port'](ag.nidaq.address[1])
  status['spectrogram'].current['port'].immutable()

  return ag, status
