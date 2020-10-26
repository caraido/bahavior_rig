# let's see how much of this we can transition to opencv... perhaps all of it
# opencv allows encoding of np arrays in hevc, which we may want to do online for compression
# # note that this will potentially consume a lot of cpu resources but should be within spec
# opencv will also be very useful for online analysis

# also i think the generator function is incorrectly implemented...

import PySpin

this_cam = 0

system = PySpin.System.GetInstance()
cam_list = system.GetCameras()
cam = cam_list.GetByIndex(this_cam)

# setup camera

cam.Init()
cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_SingleFrame)
# trigger has to be off to change source
cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
cam.TriggerMode.SetValue(PySpin.TriggerMode_On)


# setup saving

def save_frame(frame=None, stop=False):
  recorder = PySpin.SpinVideo()
  option = PySpin.AVIOption()  # for uncompressed
  option.frameRate = 30  # framerate in Hz of saved file ~ realtime

  recorder.Open(f'demo_camera{this_cam}_uncompressed', option)

  yield  # the first time this function is called it will stop here

  while not stop:
    # need to optimize here ~ can we save many frames at a time? should we use a different package??
    recorder.Append(frame)
    yield

  # when we call the function with stop=True we want to close the files
  recorder.Close()


save_frame()  # opens files

# begin acq
cam.BeginAcquisition()

# acquire an image

im = cam.GetNextImage()
im.GetImageStatus()  # should have info we want to parse, see also im.IsIncomplete()
# specifically we want to know if we have any images

# we potentially want to do a lot of things with the acquired image -- how best to do this?
save_frame(this_cam, im)
im_data = im.GetData()
im.Release()


# end acq

cam.EndAcquisition()

# cleanup

save_frame(stop=True)  # close files
cam.DeInit()
del cam
cam_list.Clear()
del cam_list
system.ReleaseInstance()
del system
