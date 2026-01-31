import can

class CanHandler:
    def __init__(self, channel="virtual", interface="virtual", bitrate=1000000):
        """
        실제 하드웨어 없이 테스트용 virtual CAN 버스 생성
        """
        try:
            self.bus = can.interface.Bus(channel=channel, interface=interface, bitrate=bitrate)
            print(f"Dummy CAN bus initialized on {channel} ({interface})")
        except Exception as e:
            print(f"Error initializing dummy CAN bus: {e}")
            self.bus = None

    def __call__(self, msg):
        print(f"Received CAN message: {msg}")

    def send_message(self, can_id, data):
        if self.bus is None:
            print(f"Dummy send: ID={can_id}, data={data}")
            return

        try:
            message = can.Message(arbitration_id=can_id, data=data, is_extended_id=False)
            self.bus.send(message)
            # print(f"Sent CAN message: ID={can_id}, data={data}")
        except can.CanError as e:
            print(f"Error sending CAN message: {e}")

    def receive_message(self, timeout=1.0):
        if self.bus is None:
            print("Dummy receive: No real messages")
            return None

        try:
            response = self.bus.recv(timeout)
            if response:
                print(f"Received CAN message: {response}")
                return response
            else:
                print("No response received within the timeout period.")
                return None
        except Exception as e:
            print(f"Error receiving CAN message: {e}")

    def close(self):
        if self.bus is None:
            print("Dummy CAN bus closed")
            return

        try:
            self.bus.shutdown()
            print("CAN bus shut down properly.")
        except Exception as e:
            print(f"Error during CAN bus shutdown: {e}")
