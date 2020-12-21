# Code Rebase 12/14

## Other todos:
- [ ] Zach: look into selecting files/path from gui
- [ ] Zach: look into checking if gpu available
- [ ] Zach: popup window when trying to close gui? to stop experiment; also displaying should turn false whenever gui is closed?
- [ ] Alec: instead of saving in Documents, save to the second SSD (e.g. D:\\behavior) and make a desktop shortcut; incorporate the rootfilename from the database
- [ ] Zach: come up with a first pass at the processinggroup
- [ ] Both: go back over some things and add warnings, but we will just pass the warnings to the gui, rather than Warnin() which goes to the python console
- [ ] remove unused imports
- [ ] import * file

## index.html
- [x] filepath should be a post method, not get

## other gui
- [ ] add axis labels to audio plot

## main.py
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

## SLCam.py
- [x] move classes to own files
- [x] make base class for cameras and nidaq to inherit from
- AcquisitionGroup
	- [ ] add a refresh pyspin.system method? for when cameras disconnect; maybe same with nidaq?
	- [ ] check if is_started before run() is executed
	- [ ] change child arrays to dicts
	- [ ] do we want a AcquisitionGroup.setFilePath() method?
		- [ ] how to handle existing filepath?
- Camera
  - [x] create a self.sleep method that multiple methods can use to sleep for correct frame interval
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
	- [x] display()
		- [x] needs _display lock
		- [x] add frame annotation in predisplay()
	- [x] Zach: __del__ may have an issue with run() race condition
	- [ ] Alec: dlc
		- [x] new thread in AcquisitionGroup, and logic to handle whether dlc is on, which camera, etc.
		- [x] 'trace' button should call acquisitiongroup function to start dlc thread
		- [ ] Zach: dlc thread runs a AcquisitionObject.run_processing() method which works similar to run()
		- [x] dlc will acquire a frame and write to self.pose
		- [x] predisplay() will take frame, and can call idu.draw_dots on self.pose
		- [x] move the cv2.putText into predisplay()
	- [ ] Alec: calibration
		- [ ] make sure that we're preventing calibration during dlc, saving (dlc_switch, start, and calibration_switch)
			- [ ] update: calibration is prevented during dlc...
		- [x] maybe we'll make a thread for calibration, and will have a while loop accessing self._frame_lock, copying self.frame, and feeding to intrinsic_calibration() etc. 
		- [ ] move intrinsic_calibration to Calib.process_frame(frame)? Calib object should store the text and corners, ids
			- [ ] update: see Camera.do_process(). Calib.do_process() or whatever should return a results dict
		- [x] Camera.predisplay() will call Calib.draw_on_frame(frame)
	- [x] update to inherit from AcquisitionObject
- Nidaq
	- [ ] method(s) to update display settings
	- [x] frame_bytes should be constructed in generator
	- [x] Zach: update start, stop, del, running, capture, save, etc. methods to conform to new Camera methods
	- [ ] log_mode
		- [x] add logic so that if display is true and filepath is none then self.log_mode should be LoggingMode.OFF?
		- [ ] test this
	- [x] remove saving/display switch workarounds
  - [x] Zach: move task creation to __init__ but keep some stuff in start when the values might change e.g. readRate/readPeriod. Then in stop() we will call nidaqmx.task.stop() instead of .close()
	- [x] copy self.sleep() over from Camera
	- [ ] add audio parameter to flip the y axis 
  - [ ] look into a better sample frequency


## still to look into:
- [x] Nidaq class
- [ ] utilities folder
- [ ] probably not calib and board classes 
- [x] talk about ProcessingGroup