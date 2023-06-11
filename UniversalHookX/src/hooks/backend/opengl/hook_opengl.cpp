#include "../../../backend.hpp"
#include "../../../console/console.hpp"

#ifdef ENABLE_BACKEND_OPENGL
#include <Windows.h>

#include <memory>

#include "hook_opengl.hpp"

#include "../../../dependencies/imgui/imgui_impl_opengl3.h"
#include "../../../dependencies/imgui/imgui_impl_win32.h"
#include "../../../dependencies/minhook/MinHook.h"

#include "../../hooks.hpp"

#include "../../../menu/menu.hpp"

#include "GLFW/glfw3.h"

#include "Postprocessing/Postprocessing.cuh"


#pragma comment(lib,"opengl32.lib")
//#pragma comment(lib,"glu32.lib")
#pragma comment(lib,"glfw3.lib")
#pragma comment(lib,"Postprocessing.lib")

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_RGB_INTEGER 0x8D98



//----- ½Ã°£ °ü·Ã -----
float Time::last_frameTime = 0.0f;
float Time::deltaTime = 0.0f;
float Time::fps_accTime = 0.0;
int Time::fps = 0;
int Time::last_fps = 0;



static std::add_pointer_t<BOOL WINAPI(HDC)> oWglSwapBuffers;

bool isInitGlfw = false;



uchar3 outputData[WIDTH * HEIGHT] = {};
uchar3 screenData[WIDTH * HEIGHT];

OperationMode operationMode = OperationMode::CPU;
Filter filter = Filter::None;

static BOOL WINAPI hkWglSwapBuffers(HDC Hdc) {
    static HGLRC oldContext = wglGetCurrentContext();
    static HGLRC newContext = wglCreateContext(Hdc);
    Time::time_update();
    if (!isInitGlfw)
    {
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        newContext = wglCreateContext(Hdc);
        wglMakeCurrent(Hdc, newContext);
        //glGenTextures(1, &texture);
        //glBindTexture(GL_TEXTURE_2D, texture);
        //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, screenData);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        //glEnable(GL_TEXTURE_2D);
        //glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    }

    //SetupOrtho
    {
        wglMakeCurrent(Hdc, newContext);
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glPushMatrix();
        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        glViewport(0, 0, viewport[2], viewport[3]);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, viewport[2], viewport[3], 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glDisable(GL_DEPTH_TEST);
    }

    //draw ·ºÅÞ±Û
    bool isInitRect = false;
    if(isInitRect)
    {
        glColor3f(1.0, 0.0, 0.0);
        glLineWidth(30);


        glBegin(GL_QUADS);
        glVertex2i(50, 90);           // //¿ÞÂÊ ¾Æ·¡
        glVertex2i(100, 90);          // ¿À¸¥ÂÊ ¾Æ·¡
            glVertex2i(100, 150);     //    ¿À¸¥ÂÊ À§
            glVertex2i(50, 150);      //     ¿ÞÂÊ À§
        glEnd();
    }


    bool isInit1 = true;
    if (isInit1)
    {
        glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, screenData);

        //ÇÈ¼¿ µÚÁý±â
        {
            for (int i = 0; i < HEIGHT; i++)
            {
                for (int j = 0; j < WIDTH; j++)
                {
                    int original_index = i * WIDTH + j;
                    int rotated_index = (HEIGHT - 1 - i) * WIDTH + (WIDTH - 1 - j);
                    outputData[rotated_index] = screenData[original_index];
                }
            }

            for (int i = 0; i < HEIGHT; i++)
            {
                for (int j = 0; j < WIDTH; j++)
                {
                    int symmetricJ = i * WIDTH + WIDTH - j - 1;
                    int original_index = i * WIDTH + j;
                    screenData[original_index] = outputData[symmetricJ];
                }
            }
        }

      Postprocessing::set_postprocessing(screenData, outputData, operationMode,filter);


       // LOG("Pixel : %d %d %d %d %d %d\n", screenData[100].x, screenData[200].x, screenData[300].x, screenData[300].x, screenData[301].x, screenData[323].x);





        //ÇÈ¼¿ ·»´õ¸µ
        bool isReal = true;
        for (int i = 0; i < HEIGHT; i ++)
        {
            for (int j = 0; j < WIDTH; j++)
            {
                
                glBegin(GL_POINTS);
                int index = WIDTH * i + j;
                //glColor3ub(outputData[index].r, outputData[index].g, outputData[index].b);
                glColor3ub(outputData[index].x, outputData[index].y, outputData[index].z);
                glVertex2i(j, i);
                glEnd();
            }
        }
    }
    //RestoreGL
    {
        glPopMatrix();
        glPopAttrib();
    }

   
   // glReadBuffer(GL_NONE);
    wglMakeCurrent(Hdc, oldContext);

    if (!H::bShuttingDown && ImGui::GetCurrentContext()) {
        if (!ImGui::GetIO().BackendRendererUserData)
            ImGui_ImplOpenGL3_Init();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

       // Menu::Render();
        bool isRender = true;
        if(isRender)
        {
            static bool* p_open = nullptr;

            static bool no_titlebar = false;
            static bool no_scrollbar = false;
            static bool no_menu = false;
            static bool no_move = false;
            static bool no_resize = false;
            static bool no_collapse = false;
            static bool no_close = false;
            static bool no_nav = false;
            static bool no_background = false;
            static bool no_bring_to_front = false;
            static bool unsaved_document = false;

            ImGuiWindowFlags window_flags = 0;
            if (no_titlebar)        window_flags |= ImGuiWindowFlags_NoTitleBar;
            if (no_scrollbar)       window_flags |= ImGuiWindowFlags_NoScrollbar;
            if (no_menu)           window_flags |= ImGuiWindowFlags_MenuBar;
            if (no_move)            window_flags |= ImGuiWindowFlags_NoMove;
            if (no_resize)          window_flags |= ImGuiWindowFlags_NoResize;
            if (no_collapse)        window_flags |= ImGuiWindowFlags_NoCollapse;
            if (no_nav)             window_flags |= ImGuiWindowFlags_NoNav;
            if (no_background)      window_flags |= ImGuiWindowFlags_NoBackground;
            if (no_bring_to_front)  window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus;
            if (unsaved_document)   window_flags |= ImGuiWindowFlags_UnsavedDocument;
            if (no_close)           p_open = NULL; // Don't pass our bool* to Begin
            const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(ImVec2(main_viewport->WorkPos.x + 650, main_viewport->WorkPos.y + 20), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(550, 680), ImGuiCond_FirstUseEver);

            // Main body of the Demo window starts here.


            //ÄÁÅÙÃ÷ ·»´õ¸µ
            {
                if (!ImGui::Begin("Postprocessing", p_open, window_flags))
                {
                    // Early out if the window is collapsed, as an optimization.
                    ImGui::End();
                    // return;
                }
                ImGui::Text("Operation Mode");
                //ImGui::AlignTextToFramePadding();
                static int select_operationMode = 0;
                ImGui::RadioButton("CPU", &select_operationMode, 0); ImGui::SameLine();
                ImGui::RadioButton("GPU", &select_operationMode, 1);
                operationMode = static_cast<OperationMode>(select_operationMode);

                static int select_filter = 0;
                ImGui::Text("Filter");
                ImGui::RadioButton("None", &select_filter, 0); ImGui::SameLine();
                ImGui::RadioButton("Gray", &select_filter, 1); ImGui::SameLine();
                ImGui::RadioButton("Sobel", &select_filter, 2);
                filter = static_cast<Filter>(select_filter);
                ImGui::End();
            }

            //¿À¹ö·¹ÀÌ ·»´õ¸µ
            {
                static int location = 0;
                ImGuiIO& io = ImGui::GetIO();
                ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
                if (location >= 0)
                {
                    const float PAD = 10.0f;
                    const ImGuiViewport* viewport = ImGui::GetMainViewport();
                    ImVec2 work_pos = viewport->WorkPos; // Use work area to avoid menu-bar/task-bar, if any!
                    ImVec2 work_size = viewport->WorkSize;
                    ImVec2 window_pos, window_pos_pivot;
                    window_pos.x = (location & 1) ? (work_pos.x + work_size.x - PAD) : (work_pos.x + PAD);
                    window_pos.y = (location & 2) ? (work_pos.y + work_size.y - PAD) : (work_pos.y + PAD);
                    window_pos_pivot.x = (location & 1) ? 1.0f : 0.0f;
                    window_pos_pivot.y = (location & 2) ? 1.0f : 0.0f;
                    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
                    window_flags |= ImGuiWindowFlags_NoMove;
                }
                else if (location == -2)
                {
                    // Center window
                    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
                    window_flags |= ImGuiWindowFlags_NoMove;
                }
                ImGui::SetNextWindowBgAlpha(0.35f); // Transparent background
                if (ImGui::Begin("Postprocessing overlay", p_open, window_flags))
                {
                    ImGui::Text("Postprocessing overlay");
                    ImGui::Text("FPS: %d\n", Time::last_fps);
                    //ImGui::Separator();
                    //if (ImGui::IsMousePosValid())
                    //    ImGui::Text("Mouse Position: (%.1f,%.1f)", io.MousePos.x, io.MousePos.y);
                    //else
                    //    ImGui::Text("Mouse Position: <invalid>");
                    if (ImGui::BeginPopupContextWindow())
                    {
                        if (ImGui::MenuItem("Custom", NULL, location == -1)) location = -1;
                        if (ImGui::MenuItem("Center", NULL, location == -2)) location = -2;
                        if (ImGui::MenuItem("Top-left", NULL, location == 0)) location = 0;
                        if (ImGui::MenuItem("Top-right", NULL, location == 1)) location = 1;
                        if (ImGui::MenuItem("Bottom-left", NULL, location == 2)) location = 2;
                        if (ImGui::MenuItem("Bottom-right", NULL, location == 3)) location = 3;
                        if (p_open && ImGui::MenuItem("Close")) *p_open = false;
                        ImGui::EndPopup();
                    }
                }
                ImGui::End();
            }
        }


        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    return oWglSwapBuffers(Hdc);
}

namespace GL {
    void Hook(HWND hwnd) {
        HMODULE openGL32 = GetModuleHandleA("opengl32.dll");
        if (openGL32) {
            LOG("[+] OpenGL32: ImageBase: 0x%p\n", openGL32);

            void* fnWglSwapBuffers = reinterpret_cast<void*>(GetProcAddress(openGL32, "wglSwapBuffers"));
            if (fnWglSwapBuffers) {
                Menu::InitializeContext(hwnd);

                // Hook
                LOG("[+] OpenGL32: fnWglSwapBuffers: 0x%p\n", fnWglSwapBuffers);
                Postprocessing::init();
                // Start Cuda-
                static MH_STATUS wsbStatus = MH_CreateHook(reinterpret_cast<void**>(fnWglSwapBuffers), &hkWglSwapBuffers, reinterpret_cast<void**>(&oWglSwapBuffers));

                MH_EnableHook(fnWglSwapBuffers);
            }
        }
    }

    void Unhook( ) {
        if (ImGui::GetCurrentContext( )) {
            if (ImGui::GetIO( ).BackendRendererUserData)
                ImGui_ImplOpenGL3_Shutdown( );

            if (ImGui::GetIO( ).BackendPlatformUserData)
                ImGui_ImplWin32_Shutdown( );

            ImGui::DestroyContext( );

            Postprocessing::release();
        }
    }
} // namespace GL
#else
#include <Windows.h>
namespace GL {
    void Hook(HWND hwnd) { LOG("[!] OpenGL backend is not enabled!\n"); }
    void Unhook( ) { }
} // namespace GL
#endif

float Time::time_acc()
{
    return 0.0f;
}

float Time::time_delta()
{
    return deltaTime;
}

void Time::time_update()
{
    float current_frameTime = (float)glfwGetTime();
    deltaTime = current_frameTime - last_frameTime;
    last_frameTime = current_frameTime;
    fps_accTime += deltaTime;
    fps++;
    if (fps_accTime > 1.f)
    {
        fps_accTime = 0.f;
        last_fps = fps;
        fps = 0;
    }
}
