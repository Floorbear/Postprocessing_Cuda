#pragma once

namespace GL {
	void Hook(HWND hwnd);
	void Unhook( );
}

class Time
{
public:
    static float time_acc(); // 프로그램 시작부터 누적
    static float time_delta(); // 프레임 사이 간격
    static void time_update();
public:
    static float last_frameTime;
    static float deltaTime;
    static float fps_accTime;
    static int fps;
    static int last_fps;
};
