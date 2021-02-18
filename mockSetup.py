# from utils.audio_settings import audio_settings
from RigStatus import RigStatus
from utils.tcp_utils import initTCP, getConnections, sendData, doShutdown
from initialStatus import initialStatus
import threading
import numpy as np
import socket
import time
import cv2

global status
global ag

global printToGUI  # TODO: should be defined in socketApp.py??
# will look something like: socketio.emit("message", f"message content string")

global annotationsToGUI  # TODO: should be defined in socketApp.py??
# will look something like socketio.emit("annotation", {"streamId":0, "data": [ {"rectangle":[(p0x, p0y), ..., (p3x, p3y)]}, ... ]})


def setup():
  class FakeAcqObj:
    def __init__(self, currStatus, address):
      self.running = False
      self.width = 1280
      self.height = 1024
      self.address = address
      if currStatus:
        self.device_serial_number = currStatus['serial number'].current
      self.socket = initTCP(address)
      self.recipients = []
      self.imarray = np.zeros((1280, 1024), dtype=np.uint8)
      self.make_frame(0)

    def display(self):
      i = 0
      while self.running:
        time.sleep(1/15)
        getConnections(self.socket, self.recipients, block=False)
        if len(self.recipients) == 0:
          continue
        else:
          sendData(self.imarray.tobytes(), self.recipients)
          i += 1
          self.make_frame(i)

    def make_frame(self, i):
      self.imarray[:] = 0
      cv2.putText(self.imarray, f'Frame {i}', (100, 100),
                  cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3, 2)

    def __del__(self):
      doShutdown(self.socket, self.recipients)

  class FakeAcqGroup:
    def __init__(self, currStatus, hostname='localhost', ports=5002):
      self.running = False
      self.nCameras = 4
      if not isinstance(ports, list):
        ports = [ports + i for i in range(5)]
      self.cameras = [FakeAcqObj(
          currStatus[f'camera {i}'].current, (hostname, ports[i])) for i in range(4)]
      self.nidaq = FakeAcqObj([], (hostname, ports[-1]))
      self.children = self.cameras + [self.nidaq]

      self.threads = []

    def stop(self):
      self.running = False
      for i in range(5):
        self.children[i].running = True
        self.threads[i].join()

    def start(self):
      self.threads = [threading.Thread(
          target=self.children[i].display) for i in range(5)]

    def run(self):
      self.running = True
      for i in range(5):
        self.children[i].running = True
        self.threads[i].start()

  status = RigStatus(initialStatus)
  ag = FakeAcqGroup(status)

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
