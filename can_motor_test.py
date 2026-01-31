import can
import time
import serial.tools.list_ports


class MotorController:
    """
    CAN 버스 통신을 관리하고 여러 모터에 명령을 전송하는 클래스.

    'with' 구문과 함께 사용하도록 설계되어 연결을 자동으로 관리합니다.

    사용 예시:
        MOTOR_1_ID = 0x141
        MOTOR_2_ID = 0x142
        with MotorController() as motor_bus:
            motor_bus.stop(MOTOR_1_ID)
            motor_bus.force_control(MOTOR_2_ID, 100)
    """

    def __init__(self, bitrate=1000000):
        """
        MotorController 인스턴스를 초기화합니다.

        :param adapter_keyword: CAN 어댑터를 찾기 위한 장치 설명 키워드
        :param bitrate: CAN 통신 속도
        """
        self.bitrate = bitrate
        self.bus = None
        self.port = None

    def _find_can_port(self):
        print("사용 가능한 COM 포트 검색 중...")
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(f"- 포트: {port.device}, 설명: {port.description}, VID: {port.vid}, PID: {port.pid}")
            # 네 어댑터의 VID/PID 매칭 (16D0:117E)
            if port.vid == 0x16D0 and port.pid == 0x117E:
                print(f"✅ CAN 어댑터를 찾았습니다: {port.device}")
                self.port = port.device
                return True
        print("❌ CAN 어댑터를 찾을 수 없습니다. 장치 연결을 확인해주세요.")
        return False

    def connect(self):
        """CAN 버스에 연결합니다."""
        if not self._find_can_port():
            return False

        try:
            self.bus = can.interface.Bus(
                interface='slcan',
                channel=self.port,
                bitrate=self.bitrate
            )
            print(f"✅ CAN 버스가 {self.port}에서 성공적으로 초기화되었습니다.")
            return True
        except Exception as e:
            print(f"❌ CAN 버스 초기화 실패: {e}")
            self.bus = None
            return False

    def disconnect(self):
        """CAN 버스와의 연결을 종료합니다."""
        if self.bus:
            self.bus.shutdown()
            print("👋 CAN 버스 연결이 종료되었습니다.")

    def __enter__(self):
        """'with' 구문 시작 시 호출됩니다."""
        if not self.connect():
            raise ConnectionError(f"Failed to connect to CAN bus on port with keyword '{self.adapter_keyword}'")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """'with' 구문 종료 시 호출됩니다."""
        self.disconnect()

    def _send_message(self, can_id, data):
        """지정된 ID로 CAN 메시지를 전송하고 응답을 수신합니다."""
        if not self.bus:
            print("❌ CAN 버스가 연결되지 않았습니다.")
            return

        try:
            message = can.Message(
                arbitration_id=can_id,
                data=data,
                is_extended_id=False
            )
            self.bus.send(message)
            response = self.bus.recv(timeout=1.0)
            print(f"To ID {hex(can_id)} -> Received Response: {response}")
        except Exception as e:
            print(f"에러: {e}")

    # --- public 모터 제어 메서드 (can_id를 인자로 받도록 수정) ---

    def stop(self, can_id):
        """지정된 ID의 모터를 정지시킵니다."""
        print(f"ID {hex(can_id)} 모터 정지 명령 전송...")
        data = [0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        self._send_message(can_id, data)

    def force_control(self, can_id, force):
        """지정된 ID 모터의 토크(힘)를 제어합니다."""
        print(f"ID {hex(can_id)} 토크 제어 명령 전송: {force}")
        force_control = int(force)
        data = [
            0xA1, 0x00, 0x00, 0x00,
            force_control & 0xFF, (force_control >> 8) & 0xFF,
            0x00, 0x00
        ]
        self._send_message(can_id, data)

    def angle_control(self, can_id, angle, max_speed):
        """지정된 ID 모터를 지정된 절대 각도로 제어합니다."""
        print(f"ID {hex(can_id)} 절대 각도 제어 명령: {angle}도, 최대 속도: {max_speed}")
        angle_control = int(angle * 10000)
        max_speed_limit = int(max_speed*10)
        data = [
            0xA4, 0x00,
            max_speed_limit & 0xFF, (max_speed_limit >> 8) & 0xFF,
            angle_control & 0xFF, (angle_control >> 8) & 0xFF,
            (angle_control >> 16) & 0xFF, (angle_control >> 24) & 0xFF
        ]
        self._send_message(can_id, data)

    def increment_angle_control(self, can_id, angle_increment, max_speed):
        """지정된 ID 모터를 지정된 증분 각도만큼 회전시킵니다."""
        print(f"ID {hex(can_id)} 증분 각도 제어 명령: {angle_increment}도, 최대 속도: {max_speed}")
        angle_increment_val = int(angle_increment * 1000)
        max_speed_limit = int(max_speed*10)
        data = [
            0xA8, 0x00,
            max_speed_limit & 0xFF, (max_speed_limit >> 8) & 0xFF,
            angle_increment_val & 0xFF, (angle_increment_val >> 8) & 0xFF,
            (angle_increment_val >> 16) & 0xFF, (angle_increment_val >> 24) & 0xFF
        ]
        self._send_message(can_id, data)

# --- 메인 실행 블록 (모터 2개 제어 예시) ---

if __name__ == "__main__":
    # 제어할 모터들의 CAN ID 정의
    MOTOR_1_ID = 0x141
    try:
        # with 구문을 사용하여 MotorController 객체 생성 및 연결
        with MotorController() as mc:

            # 1번 모터(0x141)에 토크 명령 전송
            mc.force_control(MOTOR_1_ID, 100)
            time.sleep(0.1)
            mc.force_control(MOTOR_1_ID, 30)
            time.sleep(5)
            mc.stop(MOTOR_1_ID)
            time.sleep(2)

            # 1번 모터(0x141)에 각도 증분 명령 전송
            mc.increment_angle_control(MOTOR_1_ID, -1080, 720)
            time.sleep(3)

    except ConnectionError as e:
        print(e)
    except Exception as e:
        print(f"예상치 못한 에러가 발생했습니다: {e}")