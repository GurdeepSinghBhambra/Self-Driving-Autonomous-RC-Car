/*
AUTHOR: Gurdeep Singh
PROJECT: Self Driving Prototype Vehicle 
*/

//LED 13 (in built led) for showing status
//if HIGH then, arduino is in setup()
//else, arduino is in loop()
#define LED 13


//Total Sonar Sensors Used
const unsigned int SONAR_COUNT=4;

//For Pin Connection Refer "Arduino_Pin_Connection.txt" file
//Trigger pins from Sensor 1 to 4 
unsigned int trig_pins[4] = {25, 45, 44, 24};
//Echo pins from sensor 1 to 4
unsigned int echo_pins[4] = {27, 47, 46, 26};

//Motor Controls
//Motor A with Enable A, It steers the car
#define ENA 9
//Motor B with Enable B, It accelerates the car
#define ENB 8
//Motor pins for its functions
//IN1 and IN2 belong to motor A
#define IN1 30
#define IN2 28
//IN3 and IN4 belong to motor B
#define IN3 31
#define IN4 29

//Transmit Distance Vectors; False = Transmit, True = Don't Transmit
bool NDV = false;
//If the speed of the Motor B is the same no need to alter it
unsigned int previous_speed = 0;

String getDistances()
{
  unsigned int i=0;
  double duration=-1, dist=-1;
  String distances = "";
  for(i=0; i<SONAR_COUNT; i++)
  {
    digitalWrite(trig_pins[i], LOW);
    delayMicroseconds(2);
    digitalWrite(trig_pins[i], HIGH);
    delayMicroseconds(11);
    digitalWrite(trig_pins[i], LOW);

    duration = pulseIn(echo_pins[i], HIGH, 50 * pow(10, 3)); //Wait for pulse with timeout of 1 milli seconds
    dist = (duration/2)/29.1;
    distances += String(dist, 2);
    if (i+1 != SONAR_COUNT)
    {
      distances+=",";
    }
  }
  return distances;
}

bool motor(String direct, String spd)
{
  //Serial.println("Motor: direction = "+direct+", speed = "+spd);
  unsigned int motor_speed = (spd.toInt()*255)/100;
  if(direct == "STP")
  {
    //Serial.println("\tStopping Motor");
    //analogWrite(ENA, HIGH);
    //analogWrite(ENB, HIGH);
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
  }
  else if(direct == "FWD") //Motor B is set and Motor A is reset
  {
    //Serial.println("\tForward with speed @ "+String(motor_speed)+"/255");
    if(previous_speed != motor_speed)
    { 
      previous_speed = motor_speed;
      analogWrite(ENB, motor_speed);
      analogWrite(ENA, motor_speed);
    }
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
  }
  else if(direct == "BKW") //Motor B is set and Motor A is reset
  {
    //Serial.println("\tBackward with speed @ "+String(motor_speed)+"/255");
    //analogWrite(ENA, HIGH);
    if(previous_speed != motor_speed)
    { 
      previous_speed = motor_speed;
      analogWrite(ENB, motor_speed);
      analogWrite(ENA, motor_speed);
    }
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  }
  else if(direct == "FLF") //Motor A is set and motor B is set
  {
    //Serial.println("\tForward-Left with speed @ "+String(motor_speed)+"/255");
    //analogWrite(ENA, 255);
    if(previous_speed != motor_speed)
    { 
      previous_speed = motor_speed;
      analogWrite(ENB, motor_speed);
      analogWrite(ENA, motor_speed);
    }
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
  }
  else if(direct == "FRG") //Motor A is set and motor B is set
  {
    //Serial.println("\tForward-Right with speed @ "+String(motor_speed)+"/255");
    //analogWrite(ENA, 255);
    if(previous_speed != motor_speed)
    { 
      previous_speed = motor_speed;
      analogWrite(ENB, motor_speed);
      analogWrite(ENA, motor_speed);
    }
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
  }
  else if(direct == "BLF") //Motor A is set and Motor B is set
  {
    //Serial.println("\tBackward-Left with speed @ "+String(motor_speed)+"/255");
    //analogWrite(ENA, 255);
    if(previous_speed != motor_speed)
    { 
      previous_speed = motor_speed;
      analogWrite(ENB, motor_speed);
      analogWrite(ENA, motor_speed);
    }
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  }
  else if(direct == "BRG") //Motor A is set and Motor B is set
  {
    //Serial.println("\tBackward-Right with speed @ "+String(motor_speed)+"/255");
    //analogWrite(ENA, 255);
    if(previous_speed != motor_speed)
    { 
      previous_speed = motor_speed;
      analogWrite(ENB, motor_speed);
      analogWrite(ENA, motor_speed);
    }
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  }
  else
    return false;
  return true;
}

void operate(String decision)
{
  String cmd = decision.substring(0, 3);
  if(cmd == "OFF")
  {
    analogWrite(ENA, 0);
    analogWrite(ENB, 0);
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
    NDV = false;
    Serial.println("END");
    Serial.flush();
    Serial.end();
    //setup();
  }
  else if(cmd == "NDV")
  {
    if(decision.substring(3) == "000")
      NDV = false;
    else
      NDV = true;
  }
  else if (NDV == false)
  {
    if(motor(cmd, decision.substring(3))==false)
    {
      Serial.println("ERR");
      Serial.flush();
    }
      
    Serial.println(getDistances());
    Serial.flush();
  }
  else if (NDV == true)
  {
    motor(cmd, decision.substring(3));
  }
}

void setup() {
  pinMode(LED, OUTPUT);
  digitalWrite(LED, HIGH);
  
  Serial.begin(38400);
  while(!Serial);
  
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  analogWrite(ENA, 255);
  //analogWrite(ENB, 255);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  
  delay(100);
  for(unsigned int i=0; i<SONAR_COUNT; i++)
  {
    pinMode(trig_pins[i], OUTPUT);
    pinMode(echo_pins[i], INPUT);
  }
  Serial.println("RDY");
  digitalWrite(LED, LOW);
}

void loop() {
    if(Serial.available())
    {
      operate(Serial.readStringUntil('\n'));
    }
      /*
      delay(1000);
      analogWrite(ENA, LOW);
      analogWrite(ENB, LOW);
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, LOW);
      */
}
