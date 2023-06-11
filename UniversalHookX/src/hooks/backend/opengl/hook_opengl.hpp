#pragma once

namespace GL {
	void Hook(HWND hwnd);
	void Unhook( );
}

class Time
{
public:
    static float time_acc(); // ���α׷� ���ۺ��� ����
    static float time_delta(); // ������ ���� ����
    static void time_update();
public:
    static float last_frameTime;
    static float deltaTime;
    static float fps_accTime;
    static int fps;
    static int last_fps;
};
