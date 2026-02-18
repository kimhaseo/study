import logging
import can

log = logging.getLogger(__name__)


class CanHandler:
    def __init__(self, channel="COM3", interface="slcan", bitrate=1000000):
        self.bus = can.interface.Bus(channel=channel, interface=interface, bitrate=bitrate)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def send_message(self, can_id, data):
        try:
            message = can.Message(arbitration_id=can_id, data=data, is_extended_id=False)
            self.bus.send(message)
            log.debug("Sent CAN message: ID=%s, data=%s", hex(can_id), data)
        except can.CanError as e:
            log.error("Error sending CAN message: %s", e)
            raise

    def receive_message(self, timeout=1.0):
        try:
            response = self.bus.recv(timeout)
            if response:
                log.debug("Received CAN message: %s", response)
                return response
            log.warning("No response received within timeout (%ss).", timeout)
            return None
        except Exception as e:
            log.error("Error receiving CAN message: %s", e)
            return None

    def close(self):
        try:
            self.bus.shutdown()
            log.info("CAN bus shut down properly.")
        except Exception as e:
            log.error("Error during CAN bus shutdown: %s", e)
