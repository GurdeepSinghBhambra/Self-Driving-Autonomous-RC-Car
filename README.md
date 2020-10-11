# Self-Driving-Autonomous-RC-Car
This project aims to build a self driving remote controlled car.

This project had its own dataset which was created on-site (my collage campus and my house).

Client Side Hardware:
  RC Car: 2wd car
  On board computer: Raspberry pi 3B+
  On board micro-controller: Arduino atmega 2560
  On board motor driver: L298N 2A Motor Driver Module
  On board ultrasonic sensors: 4 x HC-SR04 (At later point of time these were disabled, didn't provide much help. Still the code supports this transmission if required)
  On board power supply: 10000 mAh power bank (for raspberry pi and arduino) and 2x 9v Batteries (for motors)
  On board camera: raspberry pi v2 8mp camera

Server Side hardware:
  Just a laptop

Track:
  Lanes: black tape was used to draw the track
  Traffic light: custom traffic light was constructed using arduino.
  Obstacles: paper weigth, dummy car 

Comms Connections:
  Motor driver <-> arduino: pwm and digital pins
  Arduino <-> raspberry pi: Connected over usb connection 
  Raspberry pi <-> laptop: Connected over wifi (2 tcp ports for camera feed and commands)
  
Project Directories:
  Arduino: has the arduino files
  Car pics: pics of the car
  Client (Raspi): Has the raspberry pi files
  Data Handling modules: python modules for extractionn and preprocessing the dataset
  Dataset Sample: Sample raw images from raspberry pi
  Models: Has the mlp and nn model files
  Server (comp): has the server side files to make dataset and deploy models with auto handling feature.
