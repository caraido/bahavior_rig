# idea: 
# should not run at same time as ag


# files/directories management
# .tdms to .mat (squeaks?)
# log of errors?
# run offline dlc model on videos 
# copy to server and hard drive
# generate plots (histogram of view angle, mouse location, maybe squeak analysis down the line)

# raw and processed subfolders

# separated into threads? each runs through a sequence of tasks
  # one for video
    # run dlc models on 1 video at a time?
    # generate plots
    # sends to server, HDD
  # one for audio
    # convert to .mat / .wav etc.
    # down the line: run deepsqueak
    # down the line: analyze squeaks and plot?
    # sends to server, HDD
  # one for other stuff
    # compile errors/config info into a file in the directory
      # calibration
    # sends to server, HDD
  
  # when all the threads are done
    # delete SSD folder

