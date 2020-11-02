var cnv;
var w = 900;
var h = 600;
var ptw;
var pth;

var pps;
var fps = 1;
var speed;

var span;
var buffer;
var nFreq = 100;
var nRecv = 0; //total time points received so far
var nRecvMod = 0;

var nDrawn = 0;
var nDrawnMod = 0;

var addData = data => { //assume data is a 2d array?
  console.log('received data');
  const overflow = nRecvMod + data.s.length - buffer.length;

  if (overflow > 0) {
    console.log('overflowed');
    const lastEl = data.s.length - overflow;
    buffer.splice(nRecvMod, lastEl, ...data.s.slice(0, lastEl));
    buffer.splice(0, overflow, ...data.s.slice(lastEl));

    nRecvMod = overflow;

  } else {
    buffer.splice(nRecvMod, data.s.length, ...data.s);
    nRecvMod += data.s.length;
  }

  nRecv += data.s.length;

};

var setup = () => {
  console.log('doing setup...');

  var socket = io.connect();
  console.log(socket);

  socket.on('connect', () => console.log('connected'));
  socket.on('settings', data => {
    // center = data.center;
    console.log('got settings', data);
    span = data.fs;
    nFreq = data.nFreq;
    pps = (1 / data.window) * (1 / data.overlap) - 1; //pixels per second
    speed = -pps / fps; //pixels per frame
    buffer = new Array(nFreq * pps * 60); // store this many seconds of data

    ptw = w + speed;
    pth = h / nFreq;

    console.log('speed:', speed, 'pps: ', pps);

  });
  socket.on('fft', addData);

  cnv = createCanvas(w, h);
  cnv.parent('canvasContainer');
  noStroke();
  colorMode(HSB);
  frameRate(3 * fps); //how often to call draw()

};

var draw = () => {

  if (nRecv - nDrawn >= -speed * nFreq) {
    copy(cnv, 0, 0, w, h, speed, 0, w, h);
    let delta;
    for (var t = 0; t < -speed; t++) {
      for (var f = 0; f < nFreq; f++) {
        let c = buffer[f + nDrawnMod];
        fill(c, 255, c);
        rect(ptw + t, pth * f, -1, pth);
      }

      nDrawnMod += nFreq;
      nDrawnMod = nDrawnMod % buffer.length;
    }

    nDrawn -= speed * nFreq;
  }

};
