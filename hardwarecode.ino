#include "BluetoothSerial.h"

BluetoothSerial SerialBT;

// Motor A (Left)
const uint8_t ENA = 26;  // Speed
const uint8_t IN1 = 13;  // Direction
const uint8_t IN2 = 14;  // Direction

// Motor B (Right)
const uint8_t ENB = 25;  // Speed
const uint8_t IN3 = 16;  // Direction
const uint8_t IN4 = 27;  // Direction

// Accessories
const uint8_t BUZZER = 19;
const uint8_t TRIG = 22;
const uint8_t ECHO = 23;

const int DISTANCE_THRESHOLD = 10;  // 10cm
char currentCommand = 'S';          // Store the current active command

void setup() {
  Serial.begin(115200);
  SerialBT.begin("ESP32_Wheelchair");

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  pinMode(BUZZER, OUTPUT);
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
}

void loop() {
  int distance = getDistance();

  // Check for new commands
  // Inside your ESP32 loop()
  while (SerialBT.available()) {
    char incomingCmd = SerialBT.read();
    
    if (incomingCmd == 'H') {
      digitalWrite(BUZZER, HIGH);
      delay(200);
      digitalWrite(BUZZER, LOW);
    } else {
      currentCommand = incomingCmd;
    }
  }

  // Continuous Safety Override
  if ((distance > 0) && (distance < DISTANCE_THRESHOLD) && (currentCommand == 'F' || currentCommand == 'R' || currentCommand == 'L')) {
    driveMotors(0, 0);
  } else {
    executeCommand(currentCommand);
  }
  
  // Small delay to prevent flooding the Bluetooth connection
  delay(50); 
}

void executeCommand(char cmd) {
  switch (cmd) {
    case 'F': driveMotors(100, 100); break;
    case 'B': driveMotors(-100, -100); break;
    case 'L': driveMotors(0, 90); break;
    case 'R': driveMotors(90, 0); break;
    case 'S': driveMotors(0, 0); break;
    case 'E': driveMotors(0, 0); break; // Added 'E' for Emergency Stop
  }
}

int getDistance() {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);

  int duration = pulseIn(ECHO, HIGH);

  int distanceCm = duration * 0.034 / 2;
  return distanceCm;
}

void driveMotors(int left, int right) {
  // Left Motor (A)
  digitalWrite(IN1, left > 0);
  digitalWrite(IN2, left < 0);
  analogWrite(ENA, abs(left));

  // Right Motor (B)
  digitalWrite(IN3, right > 0);
  digitalWrite(IN4, right < 0);
  analogWrite(ENB, abs(right));
}