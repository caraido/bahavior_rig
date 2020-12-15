var cnv;
var w = window.innerWidth;
var h = 600;
var ptw;
var pth;

var pps;
var fps = 2;
var speed;
var tscaling=3; //pixels per window

var span;
var nOverlap;
var nWindow;
var buffer;
var nFreq = 100;
var nRecv = 0; //total time points received so far
var nRecvMod = 0;
var dataShape;
var nWrites;
var maxWriteSize = 20000;

var nDrawn = 0;
var nDrawnMod = 0;

//var addData = data => { //assume data is a 2d array?
//  console.log('received data', data.s.length);
//
//  buffer.splice(nRecvMod, nRecvMod + data.s.length, ...data.s)
//
//  nRecv += data.s.length;
//  nRecvMod = nRecv % buffer.length;
//
//};
//
//var addDataLooped = data => {
//    console.log('received data (loop)', data.s.length);
////    lengthRemaining = data.s.length;
//
//    let offset = 0;
//    for (let i=0; i<nWrites; i++) {
//        let nextOffset = min(offset + maxWriteSize, data.s.length);
//        let nextNRecv = nRecvMod + nextOffset;
//        buffer.splice(nRecvMod, nextNRecv, ...data.s.slice(offset, nextOffset));
//
//        offset = nextOffset;
//        nRecvMod = nextNRecv;
//    }
//    nRecv += data.s.length;
//    nRecvMod = nRecv % buffer.length;
//
////    console.log('done writing');
//}

var addDataFor = data => {
    for (var i=0; i<data.s.length; i++) {
        buffer[nRecvMod++] = data.s[i];
        buffer[nRecvMod++] = data.s[i];
        buffer[nRecvMod++] = data.s[i];
        buffer[nRecvMod++] = data.s[i];
    }
    nRecv += data.s.length;
    nRecvMod = nRecv % buffer.length;
}

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
//    pps = (1 / data.window) * (1 / data.overlap) - 1; //pixels per second
    nWindow = data.window * data.fs;
    nOverlap = nWindow * data.overlap;
    pps = Math.floor((data.fs - nOverlap)/(nWindow - nOverlap));
//
    dataShape = nFreq * pps * data.readRate * 4;

    nWrites = Math.ceil(dataShape / maxWriteSize);

//      pps = data.fs * data.window / 2 + 1;
    speed = Math.ceil(-pps / fps); //pixels per frame
    buffer = new Uint8ClampedArray(60 * dataShape);
//    buffer = new Array(60).fill(new Array(dataShape));
//    imgBuffer = new Array(60).fill(new Image(pps, h));

    ptw = w + speed*tscaling;
    pth = h / nFreq;

    console.log('speed:', speed, 'pps: ', pps, 'dataShape: ', dataShape);

    console.log('min buffer to draw: ', dataShape/fps);

//    if (nWrites == 1) {
//        socket.on('fft', addData);
//    } else {
//        socket.on('fft', addDataLooped);
//    }

    socket.on('fft',addDataFor);

  });


  cnv = createCanvas(w, h);
  cnv.parent('canvasContainer');
  noStroke();
  colorMode(HSB);
  frameRate(fps + 1); //how often to call draw()

};

var draw = () => {
  if (nRecv - nDrawn >= dataShape/fps) {
    console.log('drawing');
    loadPixels();
    const lastEl = dataShape/fps;
    for (let i = 0; i<lastEl; i++) {
        pixels[i] = buffer[nDrawnMod++];
    }
//    pixels = buffer.slice(nDrawnMod, dataShape/fps);
    updatePixels(-speed, 0);

    nDrawn += lastEl;
    nDrawnMod = nDrawn % buffer.length;

//    console.log('drawing', nRecv - nDrawn);
//    copy(cnv, 0, 0, w, h, speed * tscaling, 0, w, h);
//    let delta;
//    for (var t = 0; t < -speed; t++) {
//      for (var f = 0; f < nFreq; f++) {
//        let c = buffer[f + nDrawnMod];
//        fill(c, 255, c);
//        rect(ptw + t*tscaling, pth * f, -tscaling, pth);
//      }
//
//      nDrawnMod += nFreq;
//      nDrawnMod = nDrawnMod % buffer.length;
//    }
//
//    nDrawn -= speed * nFreq;
  }
//   else {

//    console.log('cannot draw', nRecv - nDrawn);
//  }
//  console.log('done drawing', nDrawn);

};
