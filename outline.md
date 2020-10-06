1. main python process is a flask app

2. flask app should support the following api methods:
	- start: initiates a connection to the daq, cameras, etc.
		- require username via datajoint connection?
		- spawns nidaq tasks, video tasks
		- periodically message user (over slack?) that process is running
		- start test method

	- setup: adjust recording parameters using gui

	- test: initiates tests
		- test that system is ready/capable to record
			- sufficient disk space, estimated recording time?

	- record: transitions program to data recording mode
		- opens files for writing
		- closes file transfers?
		- parse gui data (json) for database compatibility (cancelled transaction?)

	- pause: stops recording
		- closes files
		- starts file transfers?

	- stop: releases cameras, daq, etc.
		- allows for continued processing/transfer?

	- *methods should return error messages if in inappropriate state, etc.*

3. classes
  - test class:
	  - run all tests
	  -  return some data, messages, including est. recording time from disk space

  - daq class:
	  - init: setup daq for recording/testing
	  - test: validate connection, parse analog channel for trigger at desired freq, return audio, try saving
    - setup: adjust sample frequency, read frequency?
    - record: start acquisition, set file path
    - handle_chunk: run fft of data to display periodically

  - camera class:
    - init: connect to camera
    - test: validate connection, request frames via daq, try saving
    - setup: adjust framerate, exposure time, etc.
    - record: start acquisition, set file path
    - handle_chunk: 

  - tracker class:
    - nit: setup dlc, load network onto gpu?
    - test: run sample recording
    - setup:
    - record: register frames, pass frames to dlc.pose_estimation_tensorflow.nnet.predict (?), return pose estimates?

  - upload class: when recording isn't running, opens files on server and writes to them in chunks; checks for record request


		

