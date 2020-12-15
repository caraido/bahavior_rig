
start computer:
	ag = AcquisitionGroup()

open page -> preview mode
	ag.stop() # does nothing
	ag.start(filepath = None, display = True)
	ag.run()

preview mode -> record mode
	ag.stop()
	ag.start(filepath = 'blah', display = True)
	ag.run()

record mode -> preview mode
	ag.stop()
	ag.start(filepath = None, display = True)
	ag.run()

preview mode -> record mode


record mode -> stop
	ag.stop()

major bugs?:
	del ag