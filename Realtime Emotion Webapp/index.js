// Copyright (c) 2016 Muhammad Nadzeri Munawar
/*
This file is used as web server in EEG realtime emotion project.
This file uses node.js with express framework.
This file uses socket.io to communicate realtime.

See my:
- Github profile: https://github.com/nadzeri
- LinkedIn profile: https://id.linkedin.com/in/nadzeri
- Email: nadzeri.munawar94@gmail.com
*/

var express = require('express');
var app = express();
var http = require('http').Server(app);
var io = require('socket.io')(8080);

app.use(express.static('public'));

app.get('/', function(req, res){
  res.sendFile(__dirname + '/index.html');
});

//Socket to communicate realtime
io.on('connection', function(socket){
  socket.on('realtime emotion', function(msg){
    io.emit('realtime emotion', msg);
  });
});

http.listen(3000, function(){
  console.log('listening on *:3000');
});
