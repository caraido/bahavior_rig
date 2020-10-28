let main = () => {
  let spectrum = new Spectrum(
    "waterfall", {
    spectrumPercent: 20 //don't know what this does
  }
  );

  //do socketio connection
  let socket = io();
  socket.on("connect", () => console.log('connected'));
  socket.on("settings", data => {
    spectrum.setCenterHz(data.center);
    spectrum.setSpanHz(data.fs);
  });
  socket.on("fft", data => {
    spectrum.addData(data.s);
    console.log('data received');
  });

};


window.onload = main;