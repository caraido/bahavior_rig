# from setup import ag, status
from utils import path_operation_utils as pop


##serialNumbers=[19287342,19412282,17391304,17391290]

def initCallbacks(ag, status):

  def initialization(state):
    if state == 'initialized':
      ag.start()  # need the filepaths here for the temp videos?
      ag.run()
    else:
      ag.stop()

  status['initialization'].callback(initialization)

  def recording(state):
    print(f'got to recording callback with state == {state}')
    if state:
      rootfilename = status['rootfilename'].current
      print(f'rootfilename was: {rootfilename}')
      camera_list = []
      for i in range(ag.nCameras):
        camera_list.append(ag.cameras[i].device_serial_number)
      filepaths = pop.reformat_filepath('', rootfilename, camera_list)

      ag.stop()
      ag.start(filepaths=filepaths)

      ag.run()
      status['initialization'].immutable()
      status['calibration'].immutable()
      # TODO: make rootfilename and notes immutable here? and mutable below? for safety
    else:
      print('got stop message')
      ag.stop()
      ag.start()  # restart without saving
      ag.run()

      status['initialization'].mutable()
      status['calibration'].mutable()
      status['rootfilename']('')  # to make sure we don't accidentally

  status['recording'].callback(recording)

  def rootfilename(state):
    # just temporary, for debugging. want to make sure order is consistent.
    #
    print(f'attempted to set rootfilename to "{state}"')

  status['rootfilename'].callback(rootfilename)

  def notes(state):
    # TODO: should save notes under rootfile
    # status['rootfilename'].current
    print(f'attempted to update notes')
    print(state)
    print(ag.filepaths)
    pop.save_notes(state, ag.filepaths)

  status['notes'].callback(notes)

  def calibration(state):

    if state['is calibrating']:
      #TODO: start calibrating in background thread
      # state['camera serial number'].current #gives the current camera SN
      # state['type'].current == 'Intrinsic' #intrinsice or extrinsic?



      cam_id=state['camera serial number']
      cam_num=ag.camera_order.index(cam_id)
      type = state['calibration type']
      process={'mode':type}
      ag.process(cam_num,options=process)

      status['initialization'].immutable()

    else:
      ag.stop()
      ag.start()
      ag.run()
      status['initialization'].mutable()

  status['calibration'].callback(calibration)

  def spectrogram(state):
    print(f'applying new status from state: {state}')
    ag.nidaq.parse_settings(status['spectrogram'].current)
    # TODO: trying to update _nx or _nfft will cause an error
    # that means we can only update log scaling and noise correction

    # TODO: update the port number... if _nx or _nfft change

  status['spectrogram'].callback(spectrogram)

  # def camera(state):
  #cameraId = state['camera index'].current
  # if state['serial number'].current != status[f'camera {cameraId}'].current['serial number'].current:
  #  temp1 = status[f'camera {cameraId].current.copy()
  #  temp2 = status[f'camera {camera where serialNumber == requested...

  # status[f'camera {cameraId}].current = temp2
  # ...

  # for i in range(4):
  #  status[f'camera {i}'].callback(camera)
