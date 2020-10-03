import React, { Component } from 'react';

class App extends Component {


  render() {
    return (
      <div className="App">
        <p>{window.token}</p>
        <img src='/camera0'/>
        <img src='/camera1'/>
        <img src='/camera2'/>
        <img src='/camera3'/>
        <button onClick={this.onClick}>Click me</button>
      </div>
    );
  }

  onClick = () => {
    fetch('/api/', {
      method: 'POST',
      body: '',
    });
  }
  
}

export default App;
