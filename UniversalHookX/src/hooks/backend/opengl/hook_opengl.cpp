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



#pragma comment(lib,"opengl32.lib")
//#pragma comment(lib,"glu32.lib")
#pragma comment(lib,"glfw3.lib")

#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_RGB_INTEGER 0x8D98

#define WIDTH 1024
#define HEIGHT 768



static std::add_pointer_t<BOOL WINAPI(HDC)> oWglSwapBuffers;

bool isInitGlfw = false;
GLuint texture;

struct SelectionPixelIdInfo {
    unsigned char R = 0;
    unsigned char G = 0;
    unsigned char B = 0;

    //SelectionPixelIdInfo();
    //SelectionPixelIdInfo(uint32_t model_id_, uint32_t mesh_id_);
};

SelectionPixelIdInfo zeroScreenData[WIDTH * HEIGHT];
SelectionPixelIdInfo screenData[WIDTH * HEIGHT];

static BOOL WINAPI hkWglSwapBuffers(HDC Hdc) {
    static HGLRC oldContext = wglGetCurrentContext();
    static HGLRC newContext = wglCreateContext(Hdc);
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

    bool isTexture = false;
    if (isTexture)
    {
        glBindTexture(GL_TEXTURE_2D, texture);
        glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, screenData);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, screenData);

        glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex2f(0, 0);
        glTexCoord2f(1, 0);
        glVertex2f(WIDTH, 0);
        glTexCoord2f(1, 1);
        glVertex2f(WIDTH, HEIGHT);
        glTexCoord2f(0, 1);
        glVertex2f(0, HEIGHT);
        glEnd();
    }
    //draw 렉텡글
    bool isInitRect = false;
    if(isInitRect)
    {
        glColor3f(1.0, 0.0, 0.0);
        glLineWidth(30);


        glBegin(GL_QUADS);
        glVertex2i(50, 90);           // //왼쪽 아래
        glVertex2i(100, 90);          // 오른쪽 아래
            glVertex2i(100, 150);     //    오른쪽 위
            glVertex2i(50, 150);      //     왼쪽 위
        glEnd();
    }


    bool isInit1 = true;
    if (isInit1)
    {
        glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, screenData);

        //LOG("Pixel : %d %d %d %d %d %d\n", screenData[100].R, screenData[200].R, screenData[300].R, screenData[300].R, screenData[301].R, screenData[323].R);
        // Write the modified pixel data back to the frame buffer
        //glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixelData);

        //이게진짜
        bool isReal = true;
        for (int i = 0; i < HEIGHT; i ++)
        {
            for (int j = 0; j < WIDTH; j++)
            {
                
                glBegin(GL_POINTS);
                int index = WIDTH * i + j;
                /*if (screenData[index].B > 40)
                {
                    screenData[index] = { 0,0,0 };
                }*/
                glColor3ub(screenData[index].R, screenData[index].G, screenData[index].B);
                glVertex2i(j, i);
                glEnd();
                /*if (screenData[index].A < 255)
                {
                    LOG("Pixel : %d\n", screenData[index].A);
                }*/
            }
        }


        // Delete the allocated pixel data buffer
        //delete[] pixelData;

      

        //GLubyte pixelData[WIDTH * HEIGHT * 4] = { 0, }; // RGB values for each pixel
       //glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, &screenData[0]);


        //glClear(GL_COLOR_BUFFER_BIT);


       // glDrawPixels(40, 40, GL_RGBA, GL_UNSIGNED_BYTE, screenData);

    }
    //RestoreGL
    {
        glPopMatrix();
        glPopAttrib();
    }





    //glfwSwapBuffers(NULL);






    bool isInit2 = false;
    if(isInit2)
    {
        int screenWidth = 10; //Width of the screen
        int screenHeight = 10; // Height of the screen

        // Read the pixel data from the frame buffer
        GLubyte* pixelData = new GLubyte[screenWidth * screenHeight * 3]; // RGB values for each pixel
        glReadPixels(0, 0, screenWidth, screenHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixelData);
        LOG("Pixeld : %d\n", pixelData[0]);
        delete pixelData;
    }

    //{
    //    SelectionPixelIdInfo Pixel;
    //    glReadPixels(0, 0, 100, 100, GL_RGB, GL_UNSIGNED_BYTE, &Pixel);
    //    LOG("Pixel : %d %d %d\n", Pixel.R, Pixel.G, Pixel.B);
    //}
   
   // glReadBuffer(GL_NONE);
    wglMakeCurrent(Hdc, oldContext);
    if (!H::bShuttingDown && ImGui::GetCurrentContext()) {
        if (!ImGui::GetIO().BackendRendererUserData)
            ImGui_ImplOpenGL3_Init();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        Menu::Render();

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

                // Start Cuda

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
