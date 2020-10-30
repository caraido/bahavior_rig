let graph = data => {
    console.log('drawing')
    let canvas = document.getElementById('waterfall');
    let ctx = canvas.getContext('2d');
    let width = 900;
    let height = 600;
    let offset = (1 / (data.length-1)) * width

    ctx.clearRect(0,0, width, height);
    ctx.strokeRect(0,0,width, height);

    ctx.beginPath();
    ctx.moveTo(0, height - data[0]);
    for (let i=1; i<data.length; i++) {
        ctx.lineTo(i * offset, height - data[i]);
    }
    ctx.stroke()
}


let main = () => {
//  let spectrum = new Spectrum(
//    "waterfall", {
//    spectrumPercent: 20 //i think this is the percent of the canvas devoted to the current trace
//  }
//  );
//
//  window.addEventListener('keydown', e => {
//    spectrum.onKeypress(e);
//  });

  //do socketio connection
  let socket = io();
  socket.on("connect", () => console.log('connected'));
  socket.on("settings", data => {
//    spectrum.setCenterHz(data.center);
//    spectrum.setSpanHz(data.fs);
    console.log('settings', data)
  });
  socket.on("fft", data => {
    console.log('data received', data);
    graph(data.s);
//    spectrum.addData(data.s);
  });

};


window.onload = main;