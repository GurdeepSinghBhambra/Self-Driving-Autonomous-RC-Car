#define red_pin 11
#define yellow_pin 12
#define green_pin 13
#define button_pin 2

#define switch_delay 7 * 1000 //7 seconds delay
unsigned int counter = 0;
  
void manualSwitching()
{
  int curr_button_state=0;
  int prev_button_state=-1;

  curr_button_state = digitalRead(button_pin);
  if(curr_button_state != prev_button_state){
    if(curr_button_state != HIGH)
    {
      if(counter>=4) 
        counter=0;
      else
        counter++;
      if(counter == 0)
      {
        digitalWrite(red_pin, LOW);
        digitalWrite(yellow_pin, LOW);
        digitalWrite(green_pin, LOW);
      }
      else if(counter == 1)
      {
        digitalWrite(red_pin, HIGH);
        digitalWrite(yellow_pin, LOW);
        digitalWrite(green_pin, LOW);
      }
      else if(counter == 2)
      {
        digitalWrite(red_pin, LOW);
        digitalWrite(yellow_pin, HIGH);
        digitalWrite(green_pin, LOW);
      }
      else if(counter == 3)
      {
        digitalWrite(red_pin, LOW);
        digitalWrite(yellow_pin, LOW);
        digitalWrite(green_pin, HIGH);
      }
    }
    prev_button_state = curr_button_state;
  }
}

void automaticSwitching()
{
  while(true)
  {
    if(counter == 0)
    {
      digitalWrite(red_pin, HIGH);
      digitalWrite(yellow_pin, LOW);
      digitalWrite(green_pin, LOW); 
    }
    else if(counter == 1)
    {
      digitalWrite(red_pin, LOW);
      digitalWrite(yellow_pin, LOW);
      digitalWrite(green_pin, HIGH);
    }
    else if(counter == 2)
    {
      digitalWrite(red_pin, LOW);
      digitalWrite(yellow_pin, HIGH);
      digitalWrite(green_pin, LOW);
    }
    if(counter>=2) 
      counter=0;
    else
      counter++;
    delay(switch_delay);
  }
}

void setup() {
  pinMode(red_pin, OUTPUT);
  pinMode(yellow_pin, OUTPUT);
  pinMode(green_pin, OUTPUT);
  
  pinMode(button_pin, INPUT);
  digitalWrite(red_pin, LOW);
  digitalWrite(yellow_pin, LOW);
  digitalWrite(green_pin, LOW);
}  

void loop() {
delay(switch_delay);
automaticSwitching();
}
