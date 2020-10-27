let main = () => {
  let spectrum = new Spectrum(
    "waterfall", {
    spectrumPercent: 20 //don't know what this does
  }
  );

  //do socketio connection
  let socket = io();
  socket.on("connect", () => console.log('connected'));
  socket.on("settings", (data) => console.log(data));
  socket.on("fft", (data) => console.log(data));

};


window.onload = main;