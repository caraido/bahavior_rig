### Code Rebase 12/14

## Todo:
- [ ] Zach: look into selecting files/path from gui
- [ ] Zach: look into checking if gpu available
- [ ] Zach: popup window when trying to close gui? to stop experiment; also displaying should turn false whenever gui is closed?

# index.html
- [ ] filepath should be a post method, not get

# main.py
- [ ] make sure that saving and calibration are mutually exclusive, and that dlc and calibration are mutually exclusive
- [ ] move ag.run() in line 108 into its own API route
- [ ] API route to update audio settings
	- [ ] Move audio_settings into a separate config file
- [ ] Move api function definitions to separate file		
- [ ] record_switch: toggles from on to off
- [ ] dlc_switch: fix bug where model doesn't turn off
	- [ ] API route for changing models?
	- [ ] store (default) model path in config file
- [ ] calibration_switches
	- [ ] should accept data for camera_id, intrinsic vs. extrinsic, only a single api function
		- [ ] put camera names w/ serial number in config file
	- [ ] fix bug: render errors on screen, e.g. in calibration switch
	- [ ] clarify calibration switch response... should just be status OK?
- [ ] get_filepath
	- [ ] convert to camera name rather than serial number
	- [ ] return a result
- [ ] new routes:
	- [ ] come up with a list of settings for getting vs. setting
	- [ ] start/stop recording
	- [ ] start/stop acquisiton group
	- [ ] start/stop preview?
	- [ ] get/set various settings?

# SLCam.py
- [ ] move classes to own files
- [ ] make base class for cameras and nidaq to inherit from
- AcquisitionGroup
	- [ ] add a refresh pyspin.system method? for when cameras disconnect; maybe same with nidaq?
	- [ ] check if is_started before run() is executed
	- [ ] change child arrays to dicts
	- [ ] do we want a AcquisitionGroup.setFilePath() method?
		- [ ] how to handle existing filepath?
- Camera
  - [ ] create a self.sleep method that multiple methods can use to sleep for correct frame interval
  - [ ] maybe we want more camera properties? exposure for one
	- [x] ffmpeg should take size parameter from self.width and self.height, framerate?
	- [x] remove flag in run() method and just return if _running is false?
	- [x] add self._displaying_lock
	- [x] Zach: stop()
		- [x] remove dlc_switch()
		- [x] different method to turn on and off display (need to use _displaying_lock, need to make sure that we don't use 2 locks at the same time)
	- [ ] capture()
		- [x] Zach: include displaying lock
		- [x] Zach: why is frame_count inside the is_displaying block?
		- [ ] Alec: move calibration and dlc stuff out 
	- [ ] display()
		- [x] needs _display lock
		- [ ] add frame annotation here
	- [x] Zach: __del__ may have an issue with run() race condition
	- [ ] dlc:
		- [ ] new thread in AcquisitionGroup, and logic to handle whether dlc is on, which camera, etc.
		- [ ] 'trace' button should call acquisitiongroup function to start dlc thread
		- [ ] dlc thread runs a cam.run_dlc() method which works similar to display() and run(), i.e. while loop
		- [ ] dlc will acquire self.frame and write to self.pose
		- [ ] display() will take self.frame, and if self._dlc is true it will call idu.draw_dots on self.pose
		- [ ] move the cv2.putText into display()
	- [ ] calibration:
		- [ ] make sure that we're preventing calibration during dlc, saving (dlc_switch, start, and calibration_switch)
		- [ ] maybe we'll make a thread for calibration, and will have a while loop accessing self._frame_lock, copying self.frame, and feeding to intrinsic_calibration() etc. 
		- [ ] move intrinsic_calibration to Calib.process_frame(frame)? Calib object should store the text and corners, ids
		- [ ] Camera.display() will call Calib.draw_on_frame(frame)



still to look into:
Nidaq class, utilities folder, probably not calib and board classes 