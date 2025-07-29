from pythonosc import udp_client, dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer
import asyncio

class OscHandler:
    def __init__(self, config):
        # OSC 통신을 위한 client와 server의 IP와 port 설정
        self.client_ip = config['osc']['client']['ip']
        self.client_port = config['osc']['client']['port']
        self.server_ip = config['osc']['server']['ip']
        self.server_port = config['osc']['server']['port']

        # OSC 통신을 위한 client와 server 기본 객체 설정
        self.osc_client = udp_client.SimpleUDPClient(self.client_ip, self.client_port)
        self.disp = dispatcher.Dispatcher()
        self.loop = asyncio.get_event_loop()
        self.osc_server = AsyncIOOSCUDPServer((self.server_ip, self.server_port), self.disp, self.loop)

    ''' OSC 주소 맵핑 '''
    def dispatch(self, treadmillSensor, microphone, kinect, shutdown):
        # 비동기 코루틴 설정
        def wrap_async(func):
            def wrapped_func(*args):
                return asyncio.run_coroutine_threadsafe(func(*args), self.loop)
            return wrapped_func

        # 트레드밀 속도 센서 관련 주소
        self.disp.map("/start_walk", wrap_async(treadmillSensor.handle_start_walk), self.osc_client)
        self.disp.map("/end_walk", wrap_async(treadmillSensor.handle_walk_ended))

        # STT 관련 주소
        self.disp.map("/start_voice", wrap_async(microphone.handle_voice_start), self.osc_client)
        self.disp.map("/end_voice", wrap_async(microphone.handle_voice_end), self.osc_client)

        self.disp.map("/start_kinect", wrap_async(kinect.handle_kinect_start), self.osc_client)

        # 센서 클라이언트 종료 관련 주소
        self.disp.map("/shutdown_sensor", wrap_async(self.stop), self.osc_client)

        print(f"dispatch OSC")

    ''' osc server 실행 '''
    async def server_loop(self):
        transport, protocol = await self.osc_server.create_serve_endpoint()
        print(f"Listening for OSC messages on {self.server_ip}:{self.server_port}")
        try:
            # 프로그램 종료 전까지 계속해서 실행
            await asyncio.sleep(float('inf'))
        finally:
            transport.close()

    ''' osc 메시지 전송 '''
    def send_osc(self, address, data):
        self.osc_client.send_message(address, data)

    ''' osc server 종료 '''
    def stop(self):
        print("Stopping OSC Server...")
        self.loop.stop()