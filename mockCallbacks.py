from callbacks import initCallbacks as initRealCallbacks


def initCallbacks(ag, status):
  initRealCallbacks(ag, status)

  # override as needed
  status['spectrogram'].callback(lambda x: None)
