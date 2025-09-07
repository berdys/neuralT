
// main.cpp  - NeuralT - Neural Editor v3.6
#include <cfloat>
#include <cstdarg>
#include <cstddef>
#include <filesystem>
#include "main.h"
#include <cmath>

// GL & window
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <chrono>
#include <ctime>

// GLEW before GLFW
#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifdef USE_IMGUI
#include "imgui.h"
#if defined(__has_include)
    // Try common locations in order: vcpkg-style (<imgui/backends/...>),
    // vendor include (<backends/...>), then flat includes.
#  if __has_include(<imgui/backends/imgui_impl_glfw.h>)
#    include <imgui/backends/imgui_impl_glfw.h>
#    include <imgui/backends/imgui_impl_opengl3.h>
#  elif __has_include(<backends/imgui_impl_glfw.h>)
#    include <backends/imgui_impl_glfw.h>
#    include <backends/imgui_impl_opengl3.h>
#  elif __has_include(<imgui_impl_glfw.h>)
#    include <imgui_impl_glfw.h>
#    include <imgui_impl_opengl3.h>
#  else
#    error "Dear ImGui backends headers not found"
#  endif
#else
#  include <imgui_impl_glfw.h>
#  include <imgui_impl_opengl3.h>
#endif
#endif

#include <stdint.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <unordered_map>
// === Auto-inserted forward declarations to fix build on MSVC ===
struct NeuralGraph;
struct TrainState;
struct TrainSample;
double nowSeconds();
void SaveEngineConf(const char* path);
void LoadEngineConf(const char* path);
void NG_SaveJson(const NeuralGraph& g, const std::string& path);
bool NG_LoadJson(NeuralGraph& g, const std::string& path, std::string* msg);
extern TrainState gTrain;
extern std::vector<TrainSample> gDataset;
extern bool gTrainInterrupted;
// Forward decls for manual output overrides used across functions
extern std::unordered_map<int,float> gManualOutputs;
extern bool gUseManualOverrides;
// HUD state used by multiple UI sections
extern bool gHUDHasRect;
// === End auto-insert ===

// Forward decl for parameter source builder used in UI
struct Engine;
static void BuildParamSources(const Engine& E, std::vector<std::string>& names, std::vector<float>& values);

// Forward declarations for Predator dataset/training utilities (used in UI before their definitions)
static void PredatorGenerateDataset(size_t N, std::vector<TrainSample>& outDs);
static void PredatorTrainAI(NeuralGraph& net, const std::vector<TrainSample>& ds, int epochs, float lr, float l1, float l2);
static void SetupPredatorExampleGraph();

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Forward declarations
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class Engine;
extern Engine* gEngine;

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Small globals for UI
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#ifdef USE_IMGUI
static bool gShowHelp = false;
static std::string gToastMsg; static double gToastT=0.0; static bool gShowTrainingWindow=false;
static bool gEditingLabel = false; // block shortcuts while editing labels
// Visibility toggles for windows/panels (start hidden)
static bool gShowEditorWindow = false;      // F1
static bool gShowNeuralXPanel = false;      // F2 (mirrors cfg.showUI)
static bool gShowLogWindow    = false;      // F3
static bool gShowParamsWindow = false;      // F4
static bool gShowLiveFeedWindow = false;    // F6 - detachable live feed table
static bool gFullscreenPreview = false;     // Fullscreen preview toggle via button
struct UIStateBackup{ bool showUI=false, editor=false, panel=false, log=false, params=false; };
static UIStateBackup gUIBackup;
static bool gShowLoadMenu     = false;      // 'O' key or button
static bool gForceRelayoutWindows = false;  // 'Home' key re-arrangement
static bool gShowSaveMenu     = false;      // 'S' key or button
static char gSaveName[256]    = "saved/network.json";

// New: Log window
static std::vector<std::string> gLogLines;
static bool gLogAutoScroll = true;
// Result overlay state
static bool gCelebrationVisible = false;
static bool gTargetReachedVisual = false;
bool gTrainInterrupted = false;
static double gTrainDoneTime = 0.0;
// IQ tracking: accumulated across training sessions
static float gIQTotal = 0.0f;       // persists with network JSON
static float gIQSessionGain = 0.0f; // last session gain
static float gMSEStart = -1.0f;     // baseline mse at session start
static void Logf(const char* fmt, ...){
    char buf[1024];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    gLogLines.emplace_back(buf);
}
#else
// Fallback logger when IMGUI is not compiled in
static void Logf(const char* fmt, ...){
    char buf[1024];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "%s\n", buf);
}
#endif

struct WindParams {
    float strength = 150.0f; // default per request
    float speed = 1.0f;      // default per request
    float jitter = 0.13f;
};
static WindParams gWind;

static float gWindInterpolant = 0.0f;
static float gWindStrengthPhase = 0.0f;
static float gWindSpeedPhase = 0.0f;
static float gWindJitterPhase = 0.0f;

static bool gStrengthSineEnabled = false; // default: off
static bool gSpeedSineEnabled = false;    // default: off
static bool gJitterSineEnabled = true;    // default: on

// Live feed wave effect variables
static bool gLiveFeedWaveActive = false;
static float gLiveFeedWavePhase = 0.0f;
static float gLiveFeedWaveSpeed = 2.0f;
static int gLiveFeedCurrentInput = 0;
static float gLiveFeedInputDelay = 0.3f; // delay between inputs in seconds
static float gLiveFeedInputTimer = 0.0f;

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Math impl
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

static inline float deg2rad(float d){ return d * 0.017453292519943295769f; }

Mat4 Mat4::identity() {
    Mat4 r;
    r.m[0]=1; r.m[5]=1; r.m[10]=1; r.m[15]=1;
    return r;
}

Mat4 Mat4::perspective(float fovy, float aspect, float znear, float zfar) {
    float f = 1.0f / std::tan(fovy * 0.5f);
    Mat4 r{};
    r.m[0]  = f / aspect;
    r.m[5]  = f;
    r.m[10] = (zfar + znear) / (znear - zfar);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * zfar * znear) / (znear - zfar);
    return r;
}

Mat4 Mat4::ortho(float l, float ri, float b, float t, float n, float f) {
    Mat4 M = Mat4::identity();
    M.m[0]  = 2.0f/(ri-l);
    M.m[5]  = 2.0f/(t-b);
    M.m[10] = -2.0f/(f-n);
    M.m[12] = -(ri+l)/(ri-l);
    M.m[13] = -(t+b)/(t-b);
    M.m[14] = -(f+n)/(f-n);
    return M;
}

Mat4 Mat4::lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
    auto sub = [](Vec3 a, Vec3 b){ return Vec3{a.x-b.x,a.y-b.y,a.z-b.z}; };
    auto dot = [](Vec3 a, Vec3 b){ return a.x*b.x+a.y*b.y+a.z*b.z; };
    auto norm = [&](Vec3 a){ float l=std::sqrt(std::max(dot(a,a), 1e-9f)); return Vec3{a.x/l,a.y/l,a.z/l}; };
    auto cross = [](Vec3 a, Vec3 b){ return Vec3{a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; };

    Vec3 f = norm(sub(center, eye));
    Vec3 s = norm(cross(f, up));
    Vec3 u = cross(s, f);

    Mat4 r = Mat4::identity();
    r.m[0]=s.x; r.m[4]=s.y; r.m[8] =s.z;
    r.m[1]=u.x; r.m[5]=u.y; r.m[9] =u.z;
    r.m[2]=-f.x;r.m[6]=-f.y;r.m[10]=-f.z;
    r.m[12]=-(s.x*eye.x + s.y*eye.y + s.z*eye.z);
    r.m[13]=-(u.x*eye.x + u.y*eye.y + u.z*eye.z);
    r.m[14]= (f.x*eye.x + f.y*eye.y + f.z*eye.z);
    return r;
}

Mat4 Mat4::translate(const Vec3& v){ Mat4 r=Mat4::identity(); r.m[12]=v.x; r.m[13]=v.y; r.m[14]=v.z; return r; }
Mat4 Mat4::rotateY(float a){ Mat4 r=Mat4::identity(); float c=std::cos(a), s=std::sin(a); r.m[0]=c; r.m[2]=s; r.m[8]=-s; r.m[10]=c; return r; }
Mat4 Mat4::rotateX(float a){ Mat4 r=Mat4::identity(); float c=std::cos(a), s=std::sin(a); r.m[5]=c; r.m[6]=s; r.m[9]=-s; r.m[10]=c; return r; }
Mat4 Mat4::scale(const Vec3& v){ Mat4 r=Mat4::identity(); r.m[0]=v.x; r.m[5]=v.y; r.m[10]=v.z; return r; }

Mat4 Mat4::operator*(const Mat4& o) const {
    Mat4 r{};
    for(int c=0;c<4;c++){
        for(int rI=0;rI<4;rI++){
            r.m[c*4 + rI] = m[0*4 + rI]*o.m[c*4 + 0] +
                            m[1*4 + rI]*o.m[c*4 + 1] +
                            m[2*4 + rI]*o.m[c*4 + 2] +
                            m[3*4 + rI]*o.m[c*4 + 3];
        }
    }
    return r;
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Utils
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
void Timer::update(double currentTime){ now=currentTime; dt=now-last; last=now; }

void Bounds3::expand(const Vec3& p){
    min.x = std::min(min.x, p.x); min.y = std::min(min.y, p.y); min.z = std::min(min.z, p.z);
    max.x = std::max(max.x, p.x); max.y = std::max(max.y, p.y); max.z = std::max(max.z, p.z);
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// GL helpers
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
static GLuint glCreateShaderChecked(GLenum type, const char* src, std::string* log) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, 0x8B81/*GL_COMPILE_STATUS*/, &ok);
    if(!ok){
        GLint len=0; glGetShaderiv(s, 0x8B84/*GL_INFO_LOG_LENGTH*/, &len);
        std::string buf(len, '\0'); glGetShaderInfoLog(s, len, nullptr, buf.data());
        if(log) *log += buf;
    }
    return s;
}

bool Shader::compile(const char* vs, const char* fs, const char* gs, std::string* logOut){
    std::string log;
    GLuint v = glCreateShaderChecked(0x8B31/*GL_VERTEX_SHADER*/, vs, &log);
    GLuint f = glCreateShaderChecked(0x8B30/*GL_FRAGMENT_SHADER*/, fs, &log);
    GLuint g = 0;
    if(gs) g = glCreateShaderChecked(0x8DD9/*GL_GEOMETRY_SHADER*/, gs, &log);

    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f); if(gs) glAttachShader(p, g);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, 0x8B82/*GL_LINK_STATUS*/, &ok);
    if(!ok){
        GLint len=0; glGetProgramiv(p, 0x8B84/*GL_INFO_LOG_LENGTH*/, &len);
        std::string buf(len, '\0'); glGetProgramInfoLog(p, len, nullptr, buf.data());
        log += buf; if(logOut) *logOut = log;
        glDeleteProgram(p); glDeleteShader(v); glDeleteShader(f); if(gs) glDeleteShader(g);
        return false;
    }
    glDeleteShader(v); glDeleteShader(f); if(gs) glDeleteShader(g);
    program = p; if(logOut) *logOut = log;
    return true;
}

void Shader::use() const { glUseProgram(program); }
void Shader::destroy(){ if(program){ glDeleteProgram(program); program=0; } }
GLint Shader::uniform(const char* name) const { return glGetUniformLocation(program, name); }

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Mesh
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
void Mesh::upload(){
    if(vao==0) glGenVertexArrays(1, &vao);
    if(vbo==0) glGenBuffers(1, &vbo);
    if(ebo==0) glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(0x8892/*GL_ARRAY_BUFFER*/, vbo);
    glBufferData(0x8892, vertices.size()*sizeof(Vertex), vertices.data(), 0x88E4/*STATIC*/);
    glBindBuffer(0x8893/*GL_ELEMENT_ARRAY_BUFFER*/, ebo);
    glBufferData(0x8893, indices.size()*sizeof(uint32_t), indices.data(), 0x88E4);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, 0x1406/*GL_FLOAT*/, false, sizeof(Vertex), (void*)offsetof(Vertex,pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, 0x1406, false, sizeof(Vertex), (void*)offsetof(Vertex,normal));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, 0x1406, false, sizeof(Vertex), (void*)offsetof(Vertex,uv));
    glBindVertexArray(0);
    bounds = {};
    for(auto& v: vertices) bounds.expand(v.pos);
}
void Mesh::draw() const {
    glBindVertexArray(vao);
    glDrawElements(0x0004/*GL_TRIANGLES*/, (GLint)indices.size(), 0x1405/*UNSIGNED_INT*/, 0);
    glBindVertexArray(0);
}
void Mesh::destroy(){
    if(ebo){ glDeleteBuffers(1,&ebo); ebo=0; }
    if(vbo){ glDeleteBuffers(1,&vbo); vbo=0; }
    if(vao){ glDeleteVertexArrays(1,&vao); vao=0; }
    vertices.clear(); indices.clear();
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Procedural geometry
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
static void appendMesh(Mesh& dst, const Mesh& src){
    uint32_t base = (uint32_t)dst.vertices.size();
    dst.vertices.insert(dst.vertices.end(), src.vertices.begin(), src.vertices.end());
    for(auto i: src.indices) dst.indices.push_back(base + i);
}

// Arrow prism geometry: convex 5-gon (arrow) extruded by thickness along Y
Mesh Procedural::makeArrowPrism(float length, float width, float thickness, float headLen){
    Mesh m; m.vertices.clear(); m.indices.clear();
    float L = std::max(0.1f, length);
    float W = std::max(0.05f, width);
    float T = std::max(0.02f, thickness);
    float H = std::clamp(headLen, 0.05f, L*0.7f);
    // 2D outline in XZ, centered at origin, pointing +Z
    // Points: p0 tip, then counter-clockwise
    Vec3 p0{0,0, +L*0.5f};
    Vec3 p1{+W*0.5f, 0, +L*0.5f - H};
    Vec3 p2{+W*0.5f, 0, -L*0.5f};
    Vec3 p3{-W*0.5f, 0, -L*0.5f};
    Vec3 p4{-W*0.5f, 0, +L*0.5f - H};
    Vec3 poly[5] = {p0,p1,p2,p3,p4};
    auto addVert = [&](const Vec3& p, const Vec3& n, const Vec2& uv){ m.vertices.push_back({p,n,uv}); };
    // Top (+Y)
    uint32_t baseTop = (uint32_t)m.vertices.size();
    for(int i=0;i<5;i++){ Vec3 q = poly[i]; q.y = +T*0.5f; addVert(q, {0,1,0}, {0,0}); }
    for(int i=1;i<4;i++){ m.indices.insert(m.indices.end(), {baseTop+0u, baseTop+(uint32_t)i, baseTop+(uint32_t)(i+1)}); }
    // Bottom (-Y)
    uint32_t baseBot = (uint32_t)m.vertices.size();
    for(int i=0;i<5;i++){ Vec3 q = poly[i]; q.y = -T*0.5f; addVert(q, {0,-1,0}, {0,0}); }
    for(int i=1;i<4;i++){ m.indices.insert(m.indices.end(), {baseBot+0u, baseBot+(uint32_t)(i+1), baseBot+(uint32_t)i}); }
    // Sides
    auto normalXZ = [](Vec3 a, Vec3 b){ Vec3 e{b.x-a.x, 0, b.z-a.z}; float l=std::max(std::sqrt(e.x*e.x+e.z*e.z), 1e-9f); Vec3 t{e.x/l,0,e.z/l}; return Vec3{ -t.z, 0, t.x}; };
    auto addSide = [&](int i0, int i1){
        Vec3 a = poly[i0], b = poly[i1]; Vec3 n = normalXZ(a,b);
        Vec3 aT=a; aT.y=+T*0.5f; Vec3 bT=b; bT.y=+T*0.5f; Vec3 aB=a; aB.y=-T*0.5f; Vec3 bB=b; bB.y=-T*0.5f;
        uint32_t bi = (uint32_t)m.vertices.size();
        addVert(aT,n,{0,0}); addVert(bT,n,{1,0}); addVert(bB,n,{1,1}); addVert(aB,n,{0,1});
        m.indices.insert(m.indices.end(), {bi+0,bi+1,bi+2, bi+0,bi+2,bi+3});
    };
    for(int i=0;i<5;i++){ addSide(i, (i+1)%5); }
    m.upload();
    return m;
}

// Streamlined wedge prism (triangular ship) pointing +Z
Mesh Procedural::makeWedgePrism(float length, float width, float thickness){
    Mesh m; m.vertices.clear(); m.indices.clear();
    float L = std::max(0.2f, length);
    float W = std::max(0.1f, width);
    float T = std::max(0.05f, thickness);
    // Triangle in XZ: tip at +Z, base at -Z
    Vec3 p0{0, 0, +L*0.5f};           // tip
    Vec3 p1{-W*0.5f, 0, -L*0.5f};     // rear left
    Vec3 p2{+W*0.5f, 0, -L*0.5f};     // rear right
    auto addVert=[&](const Vec3& p, const Vec3& n){ m.vertices.push_back({p,n,{0,0}}); };
    // Top
    uint32_t top = (uint32_t)m.vertices.size();
    addVert({p0.x, +T*0.5f, p0.z}, {0,1,0});
    addVert({p1.x, +T*0.5f, p1.z}, {0,1,0});
    addVert({p2.x, +T*0.5f, p2.z}, {0,1,0});
    m.indices.insert(m.indices.end(), {top+0, top+1, top+2});
    // Bottom
    uint32_t bot = (uint32_t)m.vertices.size();
    addVert({p0.x, -T*0.5f, p0.z}, {0,-1,0});
    addVert({p1.x, -T*0.5f, p1.z}, {0,-1,0});
    addVert({p2.x, -T*0.5f, p2.z}, {0,-1,0});
    m.indices.insert(m.indices.end(), {bot+0, bot+2, bot+1}); // flipped winding
    // Sides (3 quads)
    auto addSide = [&](Vec3 a, Vec3 b){
        Vec3 nrm = { b.z - a.z, 0, -(b.x - a.x) };
        float len = std::max(1e-6f, std::sqrt(nrm.x*nrm.x + nrm.z*nrm.z));
        nrm.x/=len; nrm.z/=len;
        uint32_t base=(uint32_t)m.vertices.size();
        addVert({a.x, +T*0.5f, a.z}, nrm);
        addVert({b.x, +T*0.5f, b.z}, nrm);
        addVert({b.x, -T*0.5f, b.z}, nrm);
        addVert({a.x, -T*0.5f, a.z}, nrm);
        m.indices.insert(m.indices.end(), {base+0,base+1,base+2, base+0,base+2,base+3});
    };
    addSide(p1,p0); // left edge
    addSide(p0,p2); // right edge
    addSide(p2,p1); // rear edge
    m.upload();
    return m;
}

// Centered box with half-sizes sx/2, sy/2, sz/2
Mesh Procedural::makeBox(float sx, float sy, float sz){
    Mesh m; m.vertices.clear(); m.indices.clear();
    float hx=sx*0.5f, hy=sy*0.5f, hz=sz*0.5f;
    struct P{float x,y,z;};
    P p[8]={
        {-hx,-hy,-hz},{+hx,-hy,-hz},{+hx,+hy,-hz},{-hx,+hy,-hz}, // back
        {-hx,-hy,+hz},{+hx,-hy,+hz},{+hx,+hy,+hz},{-hx,+hy,+hz}  // front
    };
    auto add = [&](P a, P b, P c, P d, Vec3 n){
        uint32_t base=(uint32_t)m.vertices.size();
        m.vertices.push_back({{a.x,a.y,a.z},n,{0,0}});
        m.vertices.push_back({{b.x,b.y,b.z},n,{1,0}});
        m.vertices.push_back({{c.x,c.y,c.z},n,{1,1}});
        m.vertices.push_back({{d.x,d.y,d.z},n,{0,1}});
        m.indices.insert(m.indices.end(), {base+0,base+1,base+2, base+0,base+2,base+3});
    };
    add(p[0],p[1],p[2],p[3], {0,0,-1}); // back
    add(p[4],p[5],p[6],p[7], {0,0, 1}); // front
    add(p[0],p[4],p[7],p[3], {-1,0,0}); // left
    add(p[1],p[5],p[6],p[2], { 1,0,0}); // right
    add(p[3],p[2],p[6],p[7], {0, 1,0}); // top
    add(p[0],p[1],p[5],p[4], {0,-1,0}); // bottom
    m.upload();
    return m;
}

Mesh Procedural::makeCylinder(float R, float H, int seg, bool capped){
    Mesh m;
    for(int i=0;i<=seg;i++){
        float t = (float)i/seg;
        float ang = t * 2.f*3.1415926535f;
        float c=std::cos(ang), s=std::sin(ang);
        Vec3 n{c,0,s}; Vec3 p0{R*c,-H*0.5f,R*s}; Vec3 p1{R*c, H*0.5f,R*s};
        m.vertices.push_back({p0,n,{t,0}});
        m.vertices.push_back({p1,n,{t,1}});
    }
    for(int i=0;i<seg;i++){
        uint32_t i0 = i*2, i1 = i*2+1, i2 = (i+1)*2, i3 = (i+1)*2+1;
        m.indices.insert(m.indices.end(), {i0,i2,i1, i1,i2,i3});
    }
    if(capped){
        uint32_t centerIdx = (uint32_t)m.vertices.size();
        m.vertices.push_back({{0,-H*0.5f,0},{0,-1,0},{0.5f,0.5f}});
        for(int i=0;i<=seg;i++){
            float t = (float)i/seg; float ang = t * 2.f*3.1415926535f;
            float c=std::cos(ang), s=std::sin(ang);
            Vec3 p{R*c,-H*0.5f,R*s};
            m.vertices.push_back({p,{0,-1,0},{(c+1)*0.5f,(s+1)*0.5f}});
        }
        for(int i=0;i<seg;i++) m.indices.insert(m.indices.end(), {centerIdx, centerIdx+1u+i, centerIdx+1u+i+1u});
        centerIdx = (uint32_t)m.vertices.size();
        m.vertices.push_back({{0, H*0.5f,0},{0,1,0},{0.5f,0.5f}});
        for(int i=0;i<=seg;i++){
            float t = (float)i/seg; float ang = -t * 2.f*3.1415926535f;
            float c=std::cos(ang), s=std::sin(ang);
            Vec3 p{R*c, H*0.5f,R*s};
            m.vertices.push_back({p,{0,1,0},{(c+1)*0.5f,(s+1)*0.5f}});
        }
        for(int i=0;i<seg;i++) m.indices.insert(m.indices.end(), {centerIdx, centerIdx+1u+i, centerIdx+1u+i+1u});
    }
    m.upload(); return m;
}

Mesh Procedural::makeBlade(float rootW, float tipW, float L, float T, int slices){
    Mesh m; float halfT=T*0.5f;
    for(int i=0;i<=slices;i++){
        float t = (float)i/slices;
        float w = rootW*(1.0f-t) + tipW*t;
        float xL = -w*0.5f, xR =  w*0.5f;
        float y  = 0.0f; float z  = t*L;
        float twist = t * 0.6f; float c=std::cos(twist), s=std::sin(twist);
        auto rot = [&](float x, float yv){ return Vec3{ x*c + 0*yv, yv, x*(-s) }; };
        std::array<Vec3,4> corners = { rot(xL,-halfT), rot(xR,-halfT), rot(xR,halfT), rot(xL,halfT) };
        Vec3 n{0,1,0};
        m.vertices.push_back({{corners[0].x, y+corners[0].y, z+corners[0].z}, n, {0,0}});
        m.vertices.push_back({{corners[1].x, y+corners[1].y, z+corners[1].z}, n, {1,0}});
        m.vertices.push_back({{corners[2].x, y+corners[2].y, z+corners[2].z}, n, {1,1}});
        m.vertices.push_back({{corners[3].x, y+corners[3].y, z+corners[3].z}, n, {0,1}});
        uint32_t base = i*4;
        if(i>0){
            uint32_t b0 = base-4, b1 = base-3, b2 = base-2, b3 = base-1;
            uint32_t c0 = base,   c1 = base+1, c2 = base+2, c3 = base+3;
            m.indices.insert(m.indices.end(), { b0,b1,c1, b0,c1,c0, b1,b2,c2, b1,c2,c1, b2,b3,c3, b2,c3,c2, b3,b0,c0, b3,c0,c3 });
        }
    }
    m.upload(); return m;
}

Mesh Procedural::merge(const std::vector<Mesh>& meshes){ Mesh out; for(auto& m : meshes) appendMesh(out, m); out.upload(); return out; }

Mesh Procedural::makeLineList(const std::vector<Vec3>& pts, const std::vector<uint32_t>& inds){
    Mesh m; m.vertices.reserve(pts.size());
    for(auto& p: pts) m.vertices.push_back({p,{0,1,0},{0,0}});
    m.indices = inds; m.upload(); return m;
}

Mesh Procedural::makeGear(float innerR, float outerR, float toothDepth, int teeth, float width){
    (void)innerR; // inner hole not modeled in this simple gear; silence unused warning
    // Simple radial gear: alternating outer radius to form teeth, centered around origin, axis = Y
    Mesh m;
    int segPerTooth = 4; // base tooth shape resolution
    int seg = std::max(8, teeth * segPerTooth);
    float halfW = width*0.5f;
    // Build two rings (bottom/top), then side quads
    std::vector<Vec3> ringBottom, ringTop;
    ringBottom.reserve(seg); ringTop.reserve(seg);
    for(int i=0;i<seg;i++){
        float t = (float)i/(float)seg;
        float ang = t * 2.f*3.1415926535f;
        // tooth shaping: widen outer radius on tooth peaks
        float toothPhase = t * teeth;
        float frac = toothPhase - std::floor(toothPhase);
        float tooth = (frac<0.5f)? 1.0f : 0.0f; // crude square tooth
        float r = outerR + tooth * toothDepth;
        float c=std::cos(ang), s=std::sin(ang);
        ringBottom.push_back({r*c, -halfW, r*s});
        ringTop.push_back({r*c,  halfW, r*s});
    }
    // inner cylinder (hole) not modeled for simplicity; approximate normals
    auto addTri = [&](const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& n){
        uint32_t base = (uint32_t)m.vertices.size();
        m.vertices.push_back({a,n,{0,0}});
        m.vertices.push_back({b,n,{0,0}});
        m.vertices.push_back({c,n,{0,0}});
        m.indices.insert(m.indices.end(), {base, base+1, base+2});
    };
    // side quads between bottom/top rings
    for(int i=0;i<seg;i++){
        int j = (i+1)%seg;
        Vec3 a = ringBottom[i], b = ringBottom[j];
        Vec3 c = ringTop[j],    d = ringTop[i];
        // normal approx outward
        Vec3 edge{b.x-a.x, b.y-a.y, b.z-a.z};
        Vec3 up{0,1,0};
        Vec3 n{ edge.z*up.y - edge.y*up.z, edge.x*up.z - edge.z*up.x, edge.y*up.x - edge.x*up.y };
        float len = std::sqrt(n.x*n.x+n.y*n.y+n.z*n.z); if(len>1e-6f){ n.x/=len; n.y/=len; n.z/=len; }
        addTri(a,b,c,n); addTri(a,c,d,n);
    }
    m.upload(); return m;
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Camera
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Mat4 OrbitCamera::view() const {
    Vec3 eye{
        target.x + distance*std::cos(pitch)*std::sin(yaw),
        target.y + distance*std::sin(pitch),
        target.z + distance*std::cos(pitch)*std::cos(yaw)
    };
    return Mat4::lookAt(eye, target, {0,1,0});
}
void OrbitCamera::orbit(float dx, float dy){ yaw += dx*0.01f; pitch += dy*0.01f; pitch = std::max(-1.5f, std::min(1.5f, pitch)); }
void OrbitCamera::dolly(float dd){ distance = std::max(0.5f, distance * std::pow(0.95f, dd)); }
void OrbitCamera::pan(float dx, float dy){ float s = distance * 0.0015f; target.x += -dx*s; target.y +=  dy*s; }

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Theme
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
UITheme UITheme::NeonNoir(){
    UITheme t{}; t.name="NeonNoir";
    t.bg={0.01f,0.01f,0.01f}; t.panel={0.08f,0.02f,0.02f}; t.accent={1.0f,0.0f,0.0f}; t.grid={0.4f,0.1f,0.1f}; t.text={1.0f,1.0f,1.0f}; t.mesh={0.9f,0.1f,0.1f};
    t.nInput={1.0f,0.2f,0.2f}; t.nHidden={0.8f,0.1f,0.1f}; t.nOutput={1.0f,0.5f,0.5f}; t.nRing={1.0f,1.0f,0.0f}; t.nHighlight={1.0f,0.0f,0.0f};
    t.edgeStart={1.0f,0.0f,0.0f}; t.edgeEnd={0.60f,0.20f,0.85f}; t.edgeAlpha=0.95f; t.edgeBase=0.7f; t.edgeScale=1.0f; // violet end
    return t;
}
UITheme UITheme::Cyberpunk(){
    UITheme t{}; t.name="Cyberpunk";
    // Dark bluish background with neon cyan/magenta accents
    t.bg={0.04f,0.02f,0.07f}; t.panel={0.10f,0.05f,0.14f}; t.accent={0.05f,0.95f,0.95f}; t.grid={0.45f,0.55f,0.85f}; t.text={0.92f,0.96f,0.98f}; t.mesh={0.95f,0.20f,0.75f};
    t.nInput={0.10f,0.90f,0.90f}; t.nHidden={0.90f,0.25f,0.75f}; t.nOutput={0.95f,0.80f,0.20f}; t.nRing={1.0f,0.95f,0.45f}; t.nHighlight={0.95f,0.20f,0.75f};
    // Edge gradient cyan -> magenta
    t.edgeStart={0.05f,0.85f,0.95f}; t.edgeEnd={0.95f,0.20f,0.75f}; t.edgeAlpha=0.90f; t.edgeBase=0.7f; t.edgeScale=0.9f;
    return t;
}
UITheme UITheme::DeepSpace(){
    UITheme t{}; t.name="DeepSpace";
    t.bg={0.02f, 0.01f, 0.05f}; t.panel={0.1f, 0.05f, 0.15f}; t.accent={0.1f, 0.9f, 0.9f}; t.grid={0.2f, 0.1f, 0.3f}; t.text={0.9f, 0.95f, 1.0f}; t.mesh={0.6f, 0.2f, 0.8f};
    t.nInput={0.1f, 0.9f, 0.9f}; t.nHidden={0.9f, 0.2f, 0.7f}; t.nOutput={0.5f, 0.9f, 0.9f}; t.nRing={1.0f, 1.0f, 0.0f}; t.nHighlight={0.1f, 1.0f, 1.0f};
    t.edgeStart={0.1f, 0.9f, 0.9f}; t.edgeEnd={0.9f, 0.2f, 0.7f}; t.edgeAlpha=1.0f; t.edgeBase=0.8f; t.edgeScale=1.0f;
    return t;
}
UITheme UITheme::SolarFlare(){
    UITheme t{}; t.name="SolarFlare";
    t.bg={0.1f,0.05f,0.0f}; t.panel={0.2f,0.1f,0.0f}; t.accent={1.0f,0.8f,0.0f}; t.grid={0.7f,0.3f,0.0f}; t.text={1.0f,0.95f,0.9f}; t.mesh={1.0f,0.5f,0.0f};
    t.nInput={1.0f,0.9f,0.2f}; t.nHidden={1.0f,0.5f,0.0f}; t.nOutput={1.0f,0.7f,0.1f}; t.nRing={1.0f,1.0f,0.5f}; t.nHighlight={1.0f,1.0f,0.0f};
    t.edgeStart={1.0f,0.8f,0.0f}; t.edgeEnd={1.0f,0.4f,0.0f}; t.edgeAlpha=0.9f; t.edgeBase=0.8f; t.edgeScale=1.0f;
    return t;
}
UITheme UITheme::Sunset(){
    UITheme t{}; t.name="Sunset";
    t.bg={0.1f, 0.05f, 0.0f}; t.panel={0.3f, 0.15f, 0.05f}; t.accent={1.0f, 0.9f, 0.2f}; t.grid={0.5f, 0.2f, 0.1f}; t.text={1.0f, 0.95f, 0.85f}; t.mesh={1.0f, 0.5f, 0.1f};
    t.nInput={1.0f, 0.9f, 0.2f}; t.nHidden={1.0f, 0.3f, 0.1f}; t.nOutput={1.0f, 0.9f, 0.5f}; t.nRing={1.0f, 1.0f, 1.0f}; t.nHighlight={1.0f, 1.0f, 0.0f};
    t.edgeStart={1.0f, 0.9f, 0.2f}; t.edgeEnd={1.0f, 0.3f, 0.1f}; t.edgeAlpha=0.9f; t.edgeBase=0.7f; t.edgeScale=0.9f;
    return t;
}
UITheme UITheme::EmberForge(){
    UITheme t{}; t.name="EmberForge";
    t.bg={0.02f,0.02f,0.03f}; t.panel={0.15f,0.08f,0.05f}; t.accent={1.0f,0.3f,0.0f}; t.grid={0.5f,0.2f,0.1f}; t.text={1.0f,0.9f,0.85f}; t.mesh={1.0f,0.4f,0.1f};
    t.nInput={1.0f,0.6f,0.2f}; t.nHidden={1.0f,0.2f,0.0f}; t.nOutput={1.0f,0.8f,0.3f}; t.nRing={1.0f,1.0f,0.0f}; t.nHighlight={1.0f,0.5f,0.0f};
    t.edgeStart={1.0f,0.2f,0.0f}; t.edgeEnd={1.0f,0.7f,0.1f}; t.edgeAlpha=1.0f; t.edgeBase=0.7f; t.edgeScale=1.0f;
    return t;
}

UITheme UITheme::Crimson(){
    UITheme t{}; t.name="Crimson";
    t.bg={0.05f, 0.05f, 0.05f}; t.panel={0.1f, 0.1f, 0.1f}; t.accent={1.0f, 0.1f, 0.1f}; t.grid={0.3f, 0.3f, 0.3f}; t.text={0.9f, 0.9f, 0.9f}; t.mesh={0.8f, 0.8f, 0.8f};
    t.nInput={0.8f, 0.2f, 0.2f}; t.nHidden={0.6f, 0.6f, 0.6f}; t.nOutput={0.9f, 0.9f, 0.9f}; t.nRing={1.0f, 1.0f, 0.0f}; t.nHighlight={1.0f, 0.0f, 0.0f};
    t.edgeStart={0.1f, 0.1f, 0.1f}; t.edgeEnd={1.0f, 0.0f, 0.0f}; t.edgeAlpha=1.0f; t.edgeBase=0.8f; t.edgeScale=1.0f;
    return t;
}

UITheme UITheme::Ruby(){
    UITheme t{}; t.name="Ruby";
    t.bg={0.08f, 0.02f, 0.02f}; t.panel={0.15f, 0.05f, 0.05f}; t.accent={0.9f, 0.1f, 0.3f}; t.grid={0.4f, 0.15f, 0.15f}; t.text={1.0f, 0.9f, 0.9f}; t.mesh={0.85f, 0.3f, 0.3f};
    t.nInput={1.0f, 0.3f, 0.4f}; t.nHidden={0.7f, 0.2f, 0.3f}; t.nOutput={1.0f, 0.5f, 0.6f}; t.nRing={1.0f, 0.8f, 0.2f}; t.nHighlight={1.0f, 0.2f, 0.4f};
    t.edgeStart={0.8f, 0.1f, 0.2f}; t.edgeEnd={1.0f, 0.4f, 0.6f}; t.edgeAlpha=0.9f; t.edgeBase=0.7f; t.edgeScale=1.0f;
    return t;
}

UITheme UITheme::Aurora(){
    UITheme t{}; t.name="Aurora";
    t.bg={0.02f,0.03f,0.05f}; t.panel={0.06f,0.08f,0.12f}; t.accent={0.2f,1.0f,0.6f}; t.grid={0.2f,0.5f,0.4f}; t.text={0.9f,1.0f,0.95f}; t.mesh={0.3f,0.9f,0.7f};
    t.nInput={0.25f,0.95f,0.7f}; t.nHidden={0.65f,0.35f,0.9f}; t.nOutput={1.0f,0.85f,0.2f}; t.nRing={0.95f,1.0f,0.5f}; t.nHighlight={0.8f,0.4f,1.0f};
    t.edgeStart={0.2f,0.95f,0.65f}; t.edgeEnd={0.7f,0.4f,1.0f}; t.edgeAlpha=0.95f; t.edgeBase=0.7f; t.edgeScale=0.95f;
    return t;
}

UITheme UITheme::ForestMist(){
    UITheme t{}; t.name="ForestMist";
    t.bg={0.02f,0.05f,0.03f}; t.panel={0.06f,0.10f,0.08f}; t.accent={0.3f,0.9f,0.3f}; t.grid={0.25f,0.45f,0.35f}; t.text={0.9f,1.0f,0.9f}; t.mesh={0.55f,0.75f,0.5f};
    t.nInput={0.3f,0.9f,0.4f}; t.nHidden={0.2f,0.6f,0.3f}; t.nOutput={0.8f,0.95f,0.6f}; t.nRing={0.9f,0.9f,0.4f}; t.nHighlight={0.2f,0.8f,0.4f};
    t.edgeStart={0.2f,0.8f,0.4f}; t.edgeEnd={0.55f,0.27f,0.07f}; t.edgeAlpha=0.9f; t.edgeBase=0.7f; t.edgeScale=0.9f;
    return t;
}

UITheme UITheme::OceanWave(){
    UITheme t{}; t.name="OceanWave";
    t.bg={0.01f,0.02f,0.05f}; t.panel={0.05f,0.07f,0.12f}; t.accent={0.1f,0.6f,1.0f}; t.grid={0.2f,0.35f,0.7f}; t.text={0.9f,0.95f,1.0f}; t.mesh={0.2f,0.6f,0.95f};
    t.nInput={0.2f,0.7f,1.0f}; t.nHidden={0.15f,0.5f,0.9f}; t.nOutput={0.85f,0.95f,1.0f}; t.nRing={1.0f,1.0f,1.0f}; t.nHighlight={0.2f,0.7f,1.0f};
    t.edgeStart={0.1f,0.7f,1.0f}; t.edgeEnd={0.0f,0.3f,0.7f}; t.edgeAlpha=0.95f; t.edgeBase=0.75f; t.edgeScale=1.0f;
    return t;
}

UITheme UITheme::MonoSlate(){
    UITheme t{}; t.name="MonoSlate";
    t.bg={0.06f,0.07f,0.08f}; t.panel={0.12f,0.12f,0.14f}; t.accent={0.85f,0.85f,0.9f}; t.grid={0.35f,0.35f,0.4f}; t.text={0.95f,0.95f,0.98f}; t.mesh={0.85f,0.85f,0.9f};
    t.nInput={0.75f,0.80f,0.90f}; t.nHidden={0.65f,0.68f,0.75f}; t.nOutput={0.92f,0.92f,0.96f}; t.nRing={1.0f,1.0f,1.0f}; t.nHighlight={0.95f,0.95f,1.0f};
    t.edgeStart={0.7f,0.7f,0.78f}; t.edgeEnd={0.0f,0.0f,0.0f}; t.edgeAlpha=0.85f; t.edgeBase=0.65f; t.edgeScale=0.85f;
    return t;
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Turbine
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
void TurbineModel::rebuild(){
    destroy();
    // Scale up for a larger steampunk assembly
    float bodyR = params.hubRadius * 2.4f;
    float bodyH = params.hubRadius * 2.0f;
    float shaftR = params.shaftRadius * 1.4f;
    float shaftH = params.shaftLength * 1.2f;

    // Static vertical shaft
    shaft = Procedural::makeCylinder(shaftR, shaftH, 64, true);

    // Main turbine body (bigger)
    Mesh body = Procedural::makeCylinder(bodyR, bodyH, 72, true);

    // Gears: central ring gear (co-rotates)
    rotorTeeth = 40;
    // Solid-center gear disks (inner radius = 0)
    float gInner = 0.0f;
    float gOuter = bodyR * 1.15f;
    Mesh ringGear = Procedural::makeGear(gInner, gOuter, (gOuter-gInner)*0.35f, rotorTeeth, bodyH*0.35f);

    // Top rotor (propeller) at the top of the vertical shaft
    Mesh topRotor; topRotor.vertices.clear(); topRotor.indices.clear(); topRotor.vao=topRotor.vbo=topRotor.ebo=0;
    Mesh blade = Procedural::makeBlade(params.bladeRootWidth*1.2f, params.bladeTipWidth*1.2f,
                                       std::max(params.bladeLength, bodyR*1.0f), params.bladeThickness*1.2f, 16);
    int topBlades = std::clamp(params.blades, 1, 10);
    // Center rotor exactly on the vertical shaft axis (no lateral offset)
    float rOffset = 0.0f;
    float yTop = shaftH*0.5f + bodyR*0.10f;
    for(int i=0;i<topBlades;i++){
        float ang = (2.f*3.1415926535f * i)/std::max(1, topBlades);
        float c=std::cos(ang), s=std::sin(ang);
        Mesh b = blade;
        for(auto& v : b.vertices){
            float x=v.pos.x, y=v.pos.y, z=v.pos.z;
            float nx =  c*x + s*z;
            float nz = -s*x + c*z;
            if(rOffset != 0.0f) nx += rOffset;
            v.pos = {nx, y + yTop, nz};
        }
        appendMesh(topRotor, b);
    }
    topRotor.upload();

    // Compose rotating assembly: body + ring + side gears (horizontal & vertical appear coupled)
    rotor.destroy(); rotor.vertices.clear(); rotor.indices.clear(); rotor.vao=rotor.vbo=rotor.ebo=0;
    appendMesh(rotor, body);
    appendMesh(rotor, ringGear);
    appendMesh(rotor, topRotor);
    rotor.upload();

    // Outer static gear train on the left to visually mesh with ring
    outerTeeth = 52; outerTeethR = 52;
    float oInner = 0.0f; // solid center
    float oOuter = oInner*1.12f;
    // Ensure reasonable outer radius when inner is 0
    oOuter = std::max(oOuter, gOuter*1.25f);
    outerGear = Procedural::makeGear(oInner, oOuter, (oOuter-oInner)*0.35f, outerTeeth, bodyH*0.30f);
    outerGearOffset = -(gOuter + oOuter*0.9f); // left
    // Right gear (mirror)
    outerGearR = Procedural::makeGear(oInner, oOuter, (oOuter-oInner)*0.35f, outerTeethR, bodyH*0.30f);
    outerGearOffsetR = +(gOuter + oOuter*0.9f); // right

    // Steampunk pipes: multiple verticals + elbows + valves
    pipes.destroy(); pipes.vertices.clear(); pipes.indices.clear(); pipes.vao=pipes.vbo=pipes.ebo=0;
    auto addPipe = [&](float R, float H, float ox, float oy, float oz){
        Mesh p = Procedural::makeCylinder(R, H, 24, true);
        for(auto& v: p.vertices){ v.pos.x += ox; v.pos.y += oy; v.pos.z += oz; }
        appendMesh(pipes, p);
    };
    // Four main risers around body
    float pr = bodyR * 1.8f;
    for(int i=0;i<4;i++){
        float ang = i * (3.1415926535f*0.5f);
        float c=std::cos(ang), s=std::sin(ang);
        addPipe(bodyR*0.08f, bodyH*2.0f, pr*c, -bodyH*0.5f, pr*s);
        // Elbow (horizontal segment)
        addPipe(bodyR*0.08f, bodyH*0.8f, pr*c + (c>0? bodyH*0.4f : -bodyH*0.4f), 0.0f, pr*s);
    }
    // Valves near the body
    for(int i=0;i<6;i++){
        float t = (2.f*3.1415926535f*i)/6.f; float c=std::cos(t), s=std::sin(t);
        Mesh v = Procedural::makeCylinder(bodyR*0.10f, bodyR*0.15f, 20, true);
        for(auto& vx : v.vertices){ vx.pos.x += (bodyR*1.25f)*c; vx.pos.z += (bodyR*1.25f)*s; }
        appendMesh(pipes, v);
    }
    pipes.upload();

    // Steam lines (stylized) near pipe elbows and valves
    std::vector<Vec3> pts; std::vector<uint32_t> inds; pts.reserve(128); inds.reserve(256);
    auto addSteam = [&](Vec3 p, Vec3 dir, float len, int seg){
        uint32_t b = (uint32_t)pts.size();
        for(int i=0;i<=seg;i++){
            float t=(float)i/seg; float w = std::sin(t*3.14159f);
            pts.push_back({ p.x + dir.x*len*t, p.y + dir.y*len*t + 0.05f*w, p.z + dir.z*len*t });
            if(i>0){ inds.push_back(b+i-1); inds.push_back(b+i); }
        }
    };
    addSteam({pr,0,0}, {0,1,0}, bodyH*0.8f, 12);
    addSteam({-pr,0,0}, {0,1,0}, bodyH*0.7f, 12);
    addSteam({0,0,pr}, {0,1,0}, bodyH*0.6f, 12);
    addSteam({0,0,-pr}, {0,1,0}, bodyH*0.9f, 12);
    schematicLines = Procedural::makeLineList(pts, inds);
}

void TurbineModel::destroy(){ hub.destroy(); shaft.destroy(); blades.destroy(); gear.destroy(); rivets.destroy(); rotor.destroy(); outerGear.destroy(); outerGearR.destroy(); pipes.destroy(); schematicLines.destroy(); }

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Renderer
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
static const char* kMeshVS = R"GLSL(
#version 330 core
layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inNrm;
layout(location=2) in vec2 inUV;
uniform mat4 uMVP;
uniform mat4 uM;
out vec3 vN;
out vec3 vW;
void main(){
    vec4 w = uM * vec4(inPos,1.0);
    vW = w.xyz;
    vN = mat3(uM) * inNrm;
    gl_Position = uMVP * vec4(inPos,1.0);
}
)GLSL";

static const char* kMeshFS = R"GLSL(
#version 330 core
in vec3 vN;
in vec3 vW;
out vec4 outCol;
uniform vec3 uColor;
uniform float uMetallic;
uniform float uExposure;
void main(){
    vec3 N = normalize(vN);
    vec3 L = normalize(vec3(0.4,0.8,0.6));
    vec3 V = normalize(vec3(0.0,0.0,1.0));
    float ndl = max(dot(N,L), 0.0);
    float spec = pow(max(dot(reflect(-L,N),V),0.0), mix(32.0, 96.0, uMetallic));
    vec3 col = uColor * (0.2 + 0.8*ndl) + vec3(spec)*0.3;
    col = vec3(1.0) - exp(-col * uExposure);
    outCol = vec4(col, 1.0);
}
)GLSL";

static const char* kLineVS = R"GLSL(
#version 330 core
layout(location=0) in vec3 inPos;
uniform mat4 uMVP;
void main(){
    gl_Position = uMVP * vec4(inPos,1.0);
}
)GLSL";

static const char* kLineFS = R"GLSL(
#version 330 core
out vec4 outCol;
uniform vec3 uColor;
void main(){ outCol = vec4(uColor,1.0); }
)GLSL";

bool Renderer::init(const UITheme& t){
    theme = t;
    std::string log;
    if(!meshShader.compile(kMeshVS, kMeshFS, nullptr, &log)){ std::fprintf(stderr, "Mesh shader error:\n%s\n", log.c_str()); return false; }
    if(!lineShader.compile(kLineVS, kLineFS, nullptr, &log)){ std::fprintf(stderr, "Line shader error:\n%s\n", log.c_str()); return false; }

    // grid XZ
    std::vector<float> grid; const int N=20; const float S=0.5f;
    for(int i=-N;i<=N;i++){
        grid.push_back((float)i*S); grid.push_back(0.0f); grid.push_back(-N*S);
        grid.push_back((float)i*S); grid.push_back(0.0f); grid.push_back( N*S);
        grid.push_back(-N*S); grid.push_back(0.0f); grid.push_back((float)i*S);
        grid.push_back( N*S); grid.push_back(0.0f); grid.push_back((float)i*S);
    }
    glGenVertexArrays(1,&gridVao);
    glGenBuffers(1,&gridVbo);
    glBindVertexArray(gridVao);
    glBindBuffer(0x8892/*GL_ARRAY_BUFFER*/, gridVbo);
    glBufferData(0x8892, grid.size()*sizeof(float), grid.data(), 0x88E4/*STATIC*/);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,0x1406/*GL_FLOAT*/,false,0,(void*)0);
    glBindVertexArray(0);
    return true;
}

void Renderer::drawGrid(const Mat4& vp) const {
    lineShader.use();
    glUniformMatrix4fv(lineShader.uniform("uMVP"), 1, false, vp.m);
    glUniform3f(lineShader.uniform("uColor"), theme.grid.x, theme.grid.y, theme.grid.z);
    glBindVertexArray(gridVao);
    glDrawArrays(0x0001/*GL_LINES*/, 0, (20*2+1)*4);
    glBindVertexArray(0);
}

void Renderer::drawMesh(const Mesh& m, const Mat4& mvp, const Mat4& model, const Vec3& color, float metallic) const {
    meshShader.use();
    glUniformMatrix4fv(meshShader.uniform("uMVP"), 1, false, mvp.m);
    glUniformMatrix4fv(meshShader.uniform("uM"),   1, false, model.m);
    glUniform3f(meshShader.uniform("uColor"), color.x, color.y, color.z);
    glUniform1f(meshShader.uniform("uMetallic"), metallic);
    glUniform1f(meshShader.uniform("uExposure"), 1.0f);
    m.draw();
}

void Renderer::drawLines(const Mesh& lineList, const Mat4& mvp, const Vec3& color, float thickness) const {
    glLineWidth(thickness);
    lineShader.use();
    glUniformMatrix4fv(lineShader.uniform("uMVP"), 1, false, mvp.m);
    glUniform3f(lineShader.uniform("uColor"), color.x, color.y, color.z);
    glBindVertexArray(lineList.vao);
    glDrawElements(0x0001/*GL_LINES*/, (GLint)lineList.indices.size(), 0x1405/*UNSIGNED_INT*/, 0);
    glBindVertexArray(0);
}

void Renderer::drawMeshFlat(const Mesh& m, const Mat4& mvp, const Vec3& color) const {
    lineShader.use();
    glUniformMatrix4fv(lineShader.uniform("uMVP"), 1, false, mvp.m);
    glUniform3f(lineShader.uniform("uColor"), color.x, color.y, color.z);
    m.draw();
}

void Renderer::destroy(){
    meshShader.destroy();
    lineShader.destroy();
    if(gridVbo){ glDeleteBuffers(1,&gridVbo); gridVbo = 0; }
    if(gridVao){ glDeleteVertexArrays(1,&gridVao); gridVao = 0; }
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// NeuralGraph helpers
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#ifdef USE_IMGUI

// Training dataset definition - moved here to be visible by NG_Add/Remove
struct TrainSample{ std::vector<float> in; std::vector<float> out; std::string desc = "Default"; };
std::vector<TrainSample> gDataset;

static void NG_RebuildEdges(NeuralGraph& g){
    g.edges.clear();
    auto nid = [&](int idx){ return g.nodes[idx].id; };
    for(size_t L=0; L+1<g.layers.size(); ++L){
        for(int a : g.layers[L].nodeIdx){
            for(int b : g.layers[L+1].nodeIdx){
                g.edges.push_back({ nid(a), nid(b), 0.5f });
            }
        }
    }
    // Topology changed: invalidate cache and bump version
    g.cache.valid = false;
    g.topoVersion++;
    if(gEngine) gEngine->networkDirty = true;
}
static int NG_NextId(const NeuralGraph& g){
    int mx = -1; for(auto& n: g.nodes) mx = std::max(mx, n.id); return mx+1;
}
static void NG_AddNeuron(NeuralGraph& g, int layerIdx){
    if(layerIdx < 0 || layerIdx >= (int)g.layers.size()) return;
    auto& L = g.layers[layerIdx];

    // If adding to an I/O layer, pad the entire dataset
    if (L.kind == LayerKind::Input) {
        for (auto& sample : gDataset) {
            sample.in.push_back(0.0f);
        }
    } else if (L.kind == LayerKind::Output) {
        for (auto& sample : gDataset) {
            sample.out.push_back(0.0f);
        }
    }

    Neuron n;
    n.id = NG_NextId(g);
    n.isInput = (L.kind == LayerKind::Input);
    n.label = (L.kind==LayerKind::Input? "In" :
               L.kind==LayerKind::Output? "Out" : "H") + std::to_string((int)L.nodeIdx.size()+1);
    g.nodes.push_back(n);
    g.layers[layerIdx].nodeIdx.push_back((int)g.nodes.size()-1);
    g.layoutLayers(g.lastCanvasW>0?g.lastCanvasW:800.0f, g.lastCanvasH>0?g.lastCanvasH:500.0f);
    NG_RebuildEdges(g);
    // Graph changed: clear manual output overrides to avoid stale pins
    gManualOutputs.clear();
    gUseManualOverrides = false;
}
static void NG_RemoveLastNeuron(NeuralGraph& g, int layerIdx){
    if(layerIdx < 0 || layerIdx >= (int)g.layers.size()) return;
    auto& L = g.layers[layerIdx];
    if(L.nodeIdx.empty()) return;

    // If removing from an I/O layer, truncate the entire dataset
    if (L.kind == LayerKind::Input) {
        for (auto& sample : gDataset) {
            if (!sample.in.empty()) sample.in.pop_back();
        }
    } else if (L.kind == LayerKind::Output) {
        for (auto& sample : gDataset) {
            if (!sample.out.empty()) sample.out.pop_back();
        }
    }

    int absIdx = L.nodeIdx.back();
    int removedId = g.nodes[absIdx].id;
    L.nodeIdx.pop_back();

    // Erase neuron from nodes[] and fix indices in all layers
    g.nodes.erase(g.nodes.begin() + absIdx);
    for(auto& Lay : g.layers){
        for(int& idx : Lay.nodeIdx){
            if(idx > absIdx) idx -= 1;
        }
    }
    // Clear selection if we removed selected
    if(g.selected == removedId) g.selected = -1;

    // Rebuild full connections
    NG_RebuildEdges(g);
    g.layoutLayers(g.lastCanvasW>0?g.lastCanvasW:800.0f, g.lastCanvasH>0?g.lastCanvasH:500.0f);
    // Graph changed: clear manual output overrides to avoid stale pins
    gManualOutputs.clear();
    gUseManualOverrides = false;
}
// Insert a new hidden layer to the right of the current hidden layer
static void NG_InsertHiddenAfterCurrent(NeuralGraph& g){
    int li = g.currentLayer;
    if(li < 0 || li >= (int)g.layers.size()) return;
    if(g.layers[li].kind != LayerKind::Hidden) return;
    int insertPos = li + 1;
    Layer L; L.kind = LayerKind::Hidden; L.act = Act::ReLU;
    int hiddenCount = 0; for(const auto& Ly : g.layers) if(Ly.kind==LayerKind::Hidden) hiddenCount++;
    L.name = std::string("HL ") + std::to_string(hiddenCount + 1);
    L.nodeIdx.clear(); L.x0=L.y0=L.x1=L.y1=0.0f;
    if(insertPos < 0) {
        insertPos = 0;
    }
    if(insertPos > (int)g.layers.size()) {
        insertPos = (int)g.layers.size();
    }
    g.layers.insert(g.layers.begin() + insertPos, L);
    g.currentLayer = insertPos;
    // Add a single neuron to keep connectivity meaningful and rebuild edges/layout
    NG_AddNeuron(g, g.currentLayer);
}
// Build a specific multi-layer topology
static void NG_BuildTopology(NeuralGraph& g, int inN, const std::vector<int>& hidden, int outN){
    g.nodes.clear(); g.edges.clear(); g.layers.clear();
    g.selected = -1; std::memset(g.editBuf,0,sizeof(g.editBuf));
    g.currentLayer = 0; g.liveInputs = false;
    int id = 0;
    // Inputs
    {
        Layer L; L.kind = LayerKind::Input; L.name = "Inputs"; L.act = Act::Linear;
        for(int i=0;i<inN;i++){ Neuron n; n.id=id++; n.isInput=true; n.label="In"+std::to_string(i+1); n.speed=0.0f; n.amp=1.0f; g.nodes.push_back(n); L.nodeIdx.push_back((int)g.nodes.size()-1); }
        g.layers.push_back(std::move(L));
    }
    // Hidden layers
    for(size_t li=0; li<hidden.size(); ++li){
        Layer L; L.kind = LayerKind::Hidden; L.name = "HL "+std::to_string((int)li+1); L.act = Act::ReLU;
        for(int i=0;i<hidden[li]; ++i){ Neuron n; n.id=id++; n.isInput=false; n.label="H"+std::to_string((int)li+1)+"."+std::to_string(i+1); g.nodes.push_back(n); L.nodeIdx.push_back((int)g.nodes.size()-1); }
        g.layers.push_back(std::move(L));
    }
    // Outputs
    {
        Layer L; L.kind = LayerKind::Output; L.name = "Outputs"; L.act = Act::Sigmoid;
        for(int i=0;i<outN;i++){ Neuron n; n.id=id++; n.isInput=false; n.label="Out"+std::to_string(i+1); g.nodes.push_back(n); L.nodeIdx.push_back((int)g.nodes.size()-1); }
        g.layers.push_back(std::move(L));
    }
    NG_RebuildEdges(g);
    g.layoutLayers(g.lastCanvasW>0?g.lastCanvasW:800.0f, g.lastCanvasH>0?g.lastCanvasH:500.0f);
}

static void NG_BuildDinosaur(NeuralGraph& g) {
    NG_BuildTopology(g, 3, {5, 8}, 3);
    // Custom labels for T-Rex
    if (g.layers.size() >= 4) {
        // Inputs
        g.nodes[g.layers[0].nodeIdx[0]].label = "ProximityToPrey";
        g.nodes[g.layers[0].nodeIdx[1]].label = "HungerLevel";
        g.nodes[g.layers[0].nodeIdx[2]].label = "ThreatLevel";
        // Outputs
        g.nodes[g.layers[3].nodeIdx[0]].label = "Attack";
        g.nodes[g.layers[3].nodeIdx[1]].label = "Flee";
        g.nodes[g.layers[3].nodeIdx[2]].label = "Roar";
    }
}
// Extra helpers: save to JSON + randomize hidden (weights & biases)


static float ng_frand(float a, float b){ return a + (b-a) * (std::rand() / (float)RAND_MAX); }

// SFINAE helpers to get/set bias if Neuron has such field
template<typename T>
auto ng_get_bias(const T& n) -> decltype(n.bias) { return n.bias; }
[[maybe_unused]] static float ng_get_bias(...) { return 0.0f; }

template<typename T>
auto ng_set_bias(T& n, float v) -> decltype(n.bias = v, void()) { n.bias = v; }
[[maybe_unused]] static void ng_set_bias(...) { /* no bias present */ }

static void NG_RandomizeHidden(NeuralGraph& g, float wRange=1.0f, float bRange=0.5f){
    for(const auto& L : g.layers){
        if(L.kind != LayerKind::Hidden && L.kind != LayerKind::Output) continue;
        for(int idxN : L.nodeIdx){
            if(idxN < 0 || idxN >= (int)g.nodes.size()) continue;
            Neuron& n = g.nodes[idxN];
            ng_set_bias(n, ng_frand(-bRange, bRange));
            for(auto& e : g.edges){
                if(e.b == n.id){
                    e.w = ng_frand(-wRange, wRange);
                }
            }
        }
    }
    if(gEngine) gEngine->networkDirty = true;
}

/* --------------------------------------------------------------------------
   Deterministic weight assignment for visual gradient
   Sets all edge weights to a sorted ascending sequence (-1..1)
   Used by the shortcut: Left Shift + 1
   -------------------------------------------------------------------------- */
static void NG_AssignSortedWeights(NeuralGraph& g){
    if(g.edges.empty()) return;
    std::vector<Edge*> sorted;
    sorted.reserve(g.edges.size());
    for(auto& e : g.edges) sorted.push_back(&e);
    std::sort(sorted.begin(), sorted.end(), [](const Edge* e1, const Edge* e2){
        if(e1->a != e2->a) return e1->a < e2->a;
        return e1->b < e2->b;
    });
    size_t N = sorted.size();
    for(size_t i = 0; i < N; ++i){
        float t = (N > 1) ? (float)i / (float)(N - 1) : 0.0f;   // 0 .. 1
        sorted[i]->w = -1.0f + 2.0f * t;                         // -1 .. 1
    }
}

const char* ActName(Act a){
    switch(a){
        case Act::Linear:  return "Linear";
        case Act::ReLU:    return "ReLU";
        case Act::Sigmoid: return "Sigmoid";
        case Act::Tanh:    return "Tanh";
    }
    return "Linear";
}
Act NextAct(Act a){
    switch(a){
        case Act::Linear:  return Act::ReLU;
        case Act::ReLU:    return Act::Sigmoid;
        case Act::Sigmoid: return Act::Tanh;
        case Act::Tanh:    return Act::Linear;
    }
    return Act::Linear;
}

// Training params accessors (declared here, defined later)
static int gLessonsCount = 0;
static float TrainGetLR();
static float TrainGetL1();
static float TrainGetL2();
static int   TrainGetEpochs();
static float TrainGetTargetMSE();
static void  TrainApplyParams(float lr, float l1, float l2, int epochs, float targetMSE);

static void NG_SaveJson(const NeuralGraph& g, const std::string& path){
    std::ofstream out(path);
    if(!out) return;
    out << "{\n";
    out << "  \"version\": \"turbinos-graph-1\",\n";
    // Training parameters snapshot
    out << "  \"learning_rate\": " << TrainGetLR() << ",\n";
    out << "  \"l1\": " << TrainGetL1() << ",\n";
    out << "  \"l2\": " << TrainGetL2() << ",\n";
    out << "  \"epochs\": " << TrainGetEpochs() << ",\n";
    out << "  \"target_mse\": " << TrainGetTargetMSE() << ",\n";
    // Theme information
    const UITheme& theme = (gEngine && gEngine->window ? gEngine->renderer.theme : UITheme::Cyberpunk());
    out << "  \"theme\": {\n";
    out << "    \"name\": \"" << theme.name << "\",\n";
    out << "    \"bg\": [" << theme.bg.x << "," << theme.bg.y << "," << theme.bg.z << "],\n";
    out << "    \"panel\": [" << theme.panel.x << "," << theme.panel.y << "," << theme.panel.z << "],\n";
    out << "    \"accent\": [" << theme.accent.x << "," << theme.accent.y << "," << theme.accent.z << "],\n";
    out << "    \"grid\": [" << theme.grid.x << "," << theme.grid.y << "," << theme.grid.z << "],\n";
    out << "    \"text\": [" << theme.text.x << "," << theme.text.y << "," << theme.text.z << "],\n";
    out << "    \"mesh\": [" << theme.mesh.x << "," << theme.mesh.y << "," << theme.mesh.z << "],\n";
    out << "    \"nInput\": [" << theme.nInput.x << "," << theme.nInput.y << "," << theme.nInput.z << "],\n";
    out << "    \"nHidden\": [" << theme.nHidden.x << "," << theme.nHidden.y << "," << theme.nHidden.z << "],\n";
    out << "    \"nOutput\": [" << theme.nOutput.x << "," << theme.nOutput.y << "," << theme.nOutput.z << "],\n";
    out << "    \"nRing\": [" << theme.nRing.x << "," << theme.nRing.y << "," << theme.nRing.z << "],\n";
    out << "    \"nHighlight\": [" << theme.nHighlight.x << "," << theme.nHighlight.y << "," << theme.nHighlight.z << "],\n";
    out << "    \"edgeStart\": [" << theme.edgeStart.x << "," << theme.edgeStart.y << "," << theme.edgeStart.z << "],\n";
    out << "    \"edgeEnd\": [" << theme.edgeEnd.x << "," << theme.edgeEnd.y << "," << theme.edgeEnd.z << "],\n";
    out << "    \"edgeAlpha\": " << theme.edgeAlpha << ",\n";
    out << "    \"edgeBase\": " << theme.edgeBase << ",\n";
    out << "    \"edgeScale\": " << theme.edgeScale << "\n";
    out << "  },\n";
    out << "  \"nodes\": [\n";
    for(size_t i=0;i<g.nodes.size();++i){
        const auto& n = g.nodes[i];
        out << "    {\n";
        out << "      \"id\": " << n.id << ",\n";
        out << "      \"label\": \"" << n.label << "\",\n";
        out << "      \"desc\": \"" << n.desc << "\",\n";
        out << "      \"isInput\": " << (n.isInput? "true":"false") << ",\n";
        out << "      \"value\": " << n.value << ",\n";
        out << "      \"speed\": " << n.speed << ",\n";
        out << "      \"bias\": " << ng_get_bias(n) << ",\n";
        out << "      \"amp\": " << n.amp << ",\n";
        out << "      \"x\": " << n.x << ",\n";
        out << "      \"y\": " << n.y << "\n";
        out << "    }";
        if(i+1<g.nodes.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"layers\": [\n";
    for(size_t li=0; li<g.layers.size(); ++li){
        const auto& L = g.layers[li];
        const char* kind = (L.kind==LayerKind::Input? "Input" : (L.kind==LayerKind::Output? "Output" : "Hidden"));
        out << "    {\n";
        out << "      \"kind\": \"" << kind << "\",\n";
        out << "      \"name\": \"" << L.name << "\",\n";
        out << "      \"activation\": \"" << ActName(L.act) << "\",\n";
        out << "      \"node_ids\": [";
        for(size_t j=0;j<L.nodeIdx.size();++j){
            int idxN = L.nodeIdx[j];
            int id = (idxN>=0 && idxN < (int)g.nodes.size()) ? g.nodes[idxN].id : -1;
            out << id; if(j+1<L.nodeIdx.size()) out << ", ";
        }
        out << "]\n";
        out << "    }";
        if(li+1<g.layers.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"edges\": [\n";
    for(size_t ei=0; ei<g.edges.size(); ++ei){
        const auto& e = g.edges[ei];
        out << "    {\n";
        out << "      \"a\": " << e.a << ",\n";
        out << "      \"b\": " << e.b << ",\n";
        out << "      \"w\": " << e.w << "\n";
        out << "    }";
        if(ei+1<g.edges.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";
    // Embed lessons dataset
    out << "  \"lessons\": [\n";
    for(size_t i=0;i<gDataset.size();++i){
        const auto& s = gDataset[i];
        out << "    {\n";
        out << "      \"in\": [";
        for(size_t j=0;j<s.in.size();++j){ 
            out << s.in[j]; 
            if(j+1<s.in.size()) out << ", "; 
        }
        out << "],\n";
        out << "      \"out\": [";
        for(size_t j=0;j<s.out.size();++j){ 
            out << s.out[j]; 
            if(j+1<s.out.size()) out << ", "; 
        }
        out << "],\n";
        out << "      \"desc\": \"" << s.desc << "\"\n";
        out << "    }";
        if(i+1<gDataset.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

// Very simple loader matching NG_SaveJson format
static bool NG_LoadJson(NeuralGraph& g, const std::string& path, std::string* msg){
    std::ifstream in(path);
    if(!in){ if(msg) *msg = "Missing file: " + path; return false; }
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // Parse training parameters (if present)
    try{
        auto parseFloatC = [&](const char* key, float defv){
            try{
                std::regex rx(std::string("\"")+key+"\"\\s*:\\s*([-+eE0-9\\.]+)");
                std::smatch m; if(std::regex_search(s,m,rx)){
                    std::stringstream ss(m[1].str());
                    ss.imbue(std::locale::classic());
                    float v=defv; ss >> v; return v;
                }
            }catch(...){}
            return defv;
        };
        float _lr = parseFloatC("learning_rate", TrainGetLR());
        float _l1 = parseFloatC("l1", TrainGetL1());
        float _l2 = parseFloatC("l2", TrainGetL2());
    float _tm = parseFloatC("target_mse", TrainGetTargetMSE());
    TrainApplyParams(_lr, _l1, _l2, TrainGetEpochs(), _tm);
    }catch(...){}
    auto find_section = [&](const char* key)->std::string{
        size_t ks = s.find(std::string("\"")+key+"\"");
        if(ks==std::string::npos) return {};
        size_t lb = s.find('[', ks);
        if(lb==std::string::npos) return {};
        int depth=0; size_t i=lb;
        for(; i<s.size(); ++i){
            char c=s[i];
            if(c=='[') depth++;
            else if(c==']'){ depth--; if(depth==0){ ++i; break; } }
        }
        return s.substr(lb, i-lb);
    };
    std::string nodesSec = find_section("nodes");
    std::string layersSec = find_section("layers");
    std::string edgesSec = find_section("edges");
    std::string lessonsSec = find_section("lessons");
    if(nodesSec.empty() || layersSec.empty()){ if(msg) *msg="Bad file (sections missing)."; return false; }

    g.nodes.clear(); g.layers.clear(); g.edges.clear();

    // Try regex with desc, fall back to legacy regex without desc
    std::regex rxNodeDesc(R"(\{[^{}]*\"id\"\s*:\s*(-?\d+)[^{}]*\"label\"\s*:\s*\"([^\"]*)\"[^{}]*\"desc\"\s*:\s*\"([^\"]*)\"[^{}]*\"isInput\"\s*:\s*(true|false)[^{}]*\"value\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"speed\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"bias\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"x\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"y\"\s*:\s*([-+eE0-9\.]+)[^{}]*\})");
    std::regex rxNodeLegacy(R"(\{[^{}]*\"id\"\s*:\s*(-?\d+)[^{}]*\"label\"\s*:\s*\"([^\"]*)\"[^{}]*\"isInput\"\s*:\s*(true|false)[^{}]*\"value\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"speed\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"bias\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"x\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"y\"\s*:\s*([-+eE0-9\.]+)[^{}]*\})");
    auto it = std::sregex_iterator(nodesSec.begin(), nodesSec.end(), rxNodeDesc);
    std::unordered_map<int, int> id2idx;
    int idx=0;
    if(it != std::sregex_iterator()){
        for(; it!=std::sregex_iterator(); ++it){
            auto m=*it; Neuron n;
            n.id = std::atoi(m[1].str().c_str());
            n.label = m[2].str();
            n.desc = m[3].str();
            n.isInput = (m[4].str()=="true");
            n.value = std::atof(m[5].str().c_str());
            n.speed = std::atof(m[6].str().c_str());
            n.bias = std::atof(m[7].str().c_str());
            n.x = std::atof(m[8].str().c_str());
            n.y = std::atof(m[9].str().c_str());
            g.nodes.push_back(n);
            id2idx[n.id] = idx++;
        }
    } else {
        auto it2 = std::sregex_iterator(nodesSec.begin(), nodesSec.end(), rxNodeLegacy);
        for(; it2!=std::sregex_iterator(); ++it2){
            auto m=*it2; Neuron n;
            n.id = std::atoi(m[1].str().c_str());
            n.label = m[2].str();
            n.isInput = (m[3].str()=="true");
            n.value = std::atof(m[4].str().c_str());
            n.speed = std::atof(m[5].str().c_str());
            n.bias = std::atof(m[6].str().c_str());
            n.x = std::atof(m[7].str().c_str());
            n.y = std::atof(m[8].str().c_str());
            g.nodes.push_back(n);
            id2idx[n.id] = idx++;
        }
    }
    if(g.nodes.empty()){ if(msg) *msg="No nodes found."; return false; }

    auto strToAct = [](const std::string& s)->Act{
        if(s=="ReLU") return Act::ReLU;
        if(s=="Tanh") return Act::Tanh;
        if(s=="Sigmoid") return Act::Sigmoid;
        return Act::Linear;
    };
    std::regex rxLayer(R"(\{\s*\"kind\"\s*:\s*\"(Input|Hidden|Output)\"\s*,\s*\"name\"\s*:\s*\"([^\"]*)\"\s*,\s*\"activation\"\s*:\s*\"([^\"]*)\"\s*,\s*\"node_ids\"\s*:\s*\[([^\]]*)\]\s*\})");
    auto itL = std::sregex_iterator(layersSec.begin(), layersSec.end(), rxLayer);
    for(; itL!=std::sregex_iterator(); ++itL){
        auto m=*itL;
        Layer L;
        std::string k = m[1].str();
        L.kind = (k=="Input")? LayerKind::Input : (k=="Output")? LayerKind::Output : LayerKind::Hidden;
        L.name = m[2].str();
        L.act = strToAct(m[3].str());
        L.nodeIdx.clear();
        std::string ids = m[4].str();
        std::stringstream ss(ids);
        std::string tok;
        while(std::getline(ss, tok, ',')){
            int id = std::atoi(tok.c_str());
            auto itid = id2idx.find(id);
            if(itid!=id2idx.end()) L.nodeIdx.push_back(itid->second);
        }
        g.layers.push_back(std::move(L));
    }
    if(g.layers.empty()){ if(msg) *msg="No layers found."; return false; }

    // Load edges if present; otherwise, rebuild full connections
    if(!edgesSec.empty()){
        std::regex rxEdge(R"(\{\s*\"a\"\s*:\s*(-?\d+)\s*,\s*\"b\"\s*:\s*(-?\d+)\s*,\s*\"w\"\s*:\s*([-+eE0-9\.]+)\s*\})");
        auto itE = std::sregex_iterator(edgesSec.begin(), edgesSec.end(), rxEdge);
        for(; itE!=std::sregex_iterator(); ++itE){
            auto m=*itE;
            Edge e; e.a = std::atoi(m[1].str().c_str()); e.b = std::atoi(m[2].str().c_str()); e.w = std::atof(m[3].str().c_str());
            g.edges.push_back(e);
        }
        if(g.edges.empty()) NG_RebuildEdges(g);
    } else {
        NG_RebuildEdges(g);
    }
    // Load lessons if present
    if(!lessonsSec.empty()){
        gDataset.clear();
        
        // Get current network IO sizes to pad older datasets
        size_t numInputs = 0;
        size_t numOutputs = 0;
        for(const auto& L : g.layers){
            if(L.kind == LayerKind::Input) numInputs = L.nodeIdx.size();
            else if(L.kind == LayerKind::Output) numOutputs = L.nodeIdx.size();
        }

        std::regex rxLesson(R"(\{[^^{}]*\"in\"\s*:\s*\[([^\]]*)\][^{}]*\"out\"\s*:\s*\[([^\]]*)\][^{}]*\})");
        auto itL = std::sregex_iterator(lessonsSec.begin(), lessonsSec.end(), rxLesson);
        auto parseVec = [&](const std::string& arr){
            std::vector<float> v; std::stringstream ss(arr); std::string tok; while(std::getline(ss,tok,',')){ try{ v.push_back(std::stof(tok)); }catch(...){} } return v; };
        for(; itL!=std::sregex_iterator(); ++itL){
            auto m=*itL; TrainSample s; s.in = parseVec(m[1].str()); s.out = parseVec(m[2].str());
            std::regex rxDesc(R"(\"desc\"\s*:\s*\"([^\"]*)\")"); std::smatch md; std::string block = m.str();
            if(std::regex_search(block, md, rxDesc)) s.desc = md[1].str(); else s.desc = "Default";
            
            // Pad old data to match current network topology
            if (s.in.size() < numInputs) {
                s.in.resize(numInputs, 0.0f);
            }
            if (s.out.size() < numOutputs) {
                s.out.resize(numOutputs, 0.0f);
            }
            
            gDataset.push_back(std::move(s));
        }
        gLessonsCount = (int)gDataset.size();
    }
    
    // Load theme if present
    try{
        std::regex rxTheme(R"(\"theme\"\s*:\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\})");
        std::smatch themeMatch;
        if(std::regex_search(s, themeMatch, rxTheme)){
            std::string themeSection = themeMatch[1].str();
            
            // Parse theme name
            std::regex rxName(R"(\"name\"\s*:\s*\"([^\"]*)\")");
            std::smatch nameMatch;
            std::string themeName = "Cyberpunk"; // default
            if(std::regex_search(themeSection, nameMatch, rxName)){
                themeName = nameMatch[1].str();
            }
            
            // Helper function to parse Vec3 arrays
            auto parseVec3 = [&](const std::string& key) -> Vec3 {
                std::regex rx(std::string("\"")+key+"\"\\s*:\\s*\\[\\s*([-+eE0-9\\.]+)\\s*,\\s*([-+eE0-9\\.]+)\\s*,\\s*([-+eE0-9\\.]+)\\s*\\]");
                std::smatch m;
                if(std::regex_search(themeSection, m, rx)){
                    try{
                        return {std::stof(m[1].str()), std::stof(m[2].str()), std::stof(m[3].str())};
                    }catch(...){}
                }
                return {0.0f, 0.0f, 0.0f};
            };
            
            // Helper function to parse float values
            auto parseFloat = [&](const std::string& key, float defVal) -> float {
                std::regex rx(std::string("\"")+key+"\"\\s*:\\s*([-+eE0-9\\.]+)");
                std::smatch m;
                if(std::regex_search(themeSection, m, rx)){
                    try{
                        return std::stof(m[1].str());
                    }catch(...){}
                }
                return defVal;
            };
            
            // Create theme from loaded data
            UITheme loadedTheme;
            loadedTheme.name = themeName;
            loadedTheme.bg = parseVec3("bg");
            loadedTheme.panel = parseVec3("panel");
            loadedTheme.accent = parseVec3("accent");
            loadedTheme.grid = parseVec3("grid");
            loadedTheme.text = parseVec3("text");
            loadedTheme.mesh = parseVec3("mesh");
            loadedTheme.nInput = parseVec3("nInput");
            loadedTheme.nHidden = parseVec3("nHidden");
            loadedTheme.nOutput = parseVec3("nOutput");
            loadedTheme.nRing = parseVec3("nRing");
            loadedTheme.nHighlight = parseVec3("nHighlight");
            loadedTheme.edgeStart = parseVec3("edgeStart");
            loadedTheme.edgeEnd = parseVec3("edgeEnd");
            loadedTheme.edgeAlpha = parseFloat("edgeAlpha", 0.85f);
            loadedTheme.edgeBase = parseFloat("edgeBase", 0.6f);
            loadedTheme.edgeScale = parseFloat("edgeScale", 0.8f);
            
            // Apply theme to renderer if engine exists
            if(gEngine && gEngine->window){
                gEngine->renderer.theme = loadedTheme;
                Logf("Theme loaded: %s", themeName.c_str());
            }
        }
    }catch(...){
        // If theme loading fails, continue with default theme
        Logf("Theme loading failed, using default theme");
    }
    // After loading, compute a fresh layout to scale to current canvas
    float W = g.lastCanvasW>0? g.lastCanvasW : 800.0f;
    float H = g.lastCanvasH>0? g.lastCanvasH : 500.0f;
    g.layoutLayers(W, H);
    for(auto& n : g.nodes){ n.vx=0.0f; n.vy=0.0f; }
    return true;
}

// ===================== Training dataset & Engine config =====================
std::unordered_map<int,float> gManualOutputs;
bool gUseManualOverrides = false;

TrainState gTrain;

// Accessors to avoid forward-dep issues
static float TrainGetLR(){ return gTrain.lr; }
static float TrainGetL1(){ return gTrain.l1; }
static float TrainGetL2(){ return gTrain.l2; }
static int   TrainGetEpochs(){ return gTrain.totalEpochs; }
static float TrainGetTargetMSE(){ return gTrain.targetMSE; }
static void  TrainApplyParams(float lr, float l1, float l2, int epochs, float targetMSE){
    gTrain.lr = lr; gTrain.l1 = l1; gTrain.l2 = l2; gTrain.totalEpochs = epochs; gTrain.targetMSE = targetMSE;
}

struct EngineConf{
    std::string engine_version;
    float lr=0.003f, l1=0.0003f, l2=0.0005f;
    int epochs=5000;
    // Camera on Turbinoes Shit Dick Machine
    float cam_yaw=0.7f, cam_pitch=0.35f, cam_distance=4.0f;
    float cam_tx=0.0f, cam_ty=0.0f, cam_tz=0.0f;
};
static EngineConf gConf;

static double nowSeconds(){ return glfwGetTime(); }

// Forward declaration to access camera when saving/loading config
extern Engine* gEngine;
static void SaveEngineConf(const char* path){
    std::ofstream f(path);
    if(!f) return;
    f << "{\n";
    f << "  \"engine_version\": \"" << kAppVersion << ",\n";

    f << "  \"learning_rate\": " << gTrain.lr << ",\n";
    f << "  \"l1\": " << gTrain.l1 << ",\n";
    f << "  \"l2\": " << gTrain.l2 << ",\n";
    f << "  \"epochs\": " << gTrain.totalEpochs << "\n";
    // Camera
    f << ",\n  \"cam_yaw\": " << gEngine->cam.yaw;
    f << ",\n  \"cam_pitch\": " << gEngine->cam.pitch;
    f << ",\n  \"cam_distance\": " << gEngine->cam.distance;
    f << ",\n  \"cam_tx\": " << gEngine->cam.target.x;
    f << ",\n  \"cam_ty\": " << gEngine->cam.target.y;
    f << ",\n  \"cam_tz\": " << gEngine->cam.target.z;
    f << "}\n";
}

static float parseNumber(const std::string& s, const char* key, float defv){
    try{
        std::regex rx(std::string("\"")+key+"\"\\s*:\\s*([-+eE0-9\\.]+)");
        std::smatch m;
        if(std::regex_search(s,m,rx)){ return std::stof(m[1].str()); }
    }catch(...){}
    return defv;
}
static int parseInt(const std::string& s, const char* key, int defv){
    try{
        std::regex rx(std::string("\"")+key+"\"\\s*:\\s*([-+0-9]+)");
        std::smatch m;
        if(std::regex_search(s,m,rx)){ return std::stoi(m[1].str()); }
    }catch(...){}
    return defv;
}

static void LoadEngineConf(const char* path){
    std::ifstream f(path);
    if(!f){
        SaveEngineConf(path);
        return;
    }
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    gTrain.lr = parseNumber(all, "learning_rate", gTrain.lr);
    gTrain.l1 = parseNumber(all, "l1", gTrain.l1);
    gTrain.l2 = parseNumber(all, "l2", gTrain.l2);
    gTrain.targetMSE = parseNumber(all, "target_mse", gTrain.targetMSE);
    gTrain.totalEpochs = parseInt(all, "epochs", gTrain.totalEpochs);
    // lessons count is derived from dataset file; do not read from engine.conf anymore
    // Camera (apply immediately if engine exists)
    float cyaw = parseNumber(all, "cam_yaw", gConf.cam_yaw);
    float cpitch = parseNumber(all, "cam_pitch", gConf.cam_pitch);
    float cdist = parseNumber(all, "cam_distance", gConf.cam_distance);
    float ctx = parseNumber(all, "cam_tx", gConf.cam_tx);
    float cty = parseNumber(all, "cam_ty", gConf.cam_ty);
    float ctz = parseNumber(all, "cam_tz", gConf.cam_tz);
    gConf.cam_yaw=cyaw; gConf.cam_pitch=cpitch; gConf.cam_distance=cdist;
    gConf.cam_tx=ctx; gConf.cam_ty=cty; gConf.cam_tz=ctz;
    if(gEngine){
        gEngine->cam.yaw = cyaw;
        gEngine->cam.pitch = cpitch;
        gEngine->cam.distance = cdist;
        gEngine->cam.target = {ctx, cty, ctz};
    }
}

// Deprecated: lessons are embedded in the saved network file; no separate lessons.json writer

static void CollectIO(const NeuralGraph& g, std::vector<float>& in, std::vector<float>& out){
    in.clear(); out.clear();
    for(const auto& L : g.layers){
        if(L.kind == LayerKind::Input){
            for(int idx : L.nodeIdx) in.push_back(g.nodes[idx].value);
        } else if(L.kind == LayerKind::Output){
            for(int idx : L.nodeIdx) out.push_back(g.nodes[idx].value);
        }
    }
}

static float ActApply(Act a, float x){
    switch(a){
        default: 
        case Act::Linear: return x;
        case Act::ReLU:   return x>0.f?x:0.f;
        case Act::Sigmoid: return 1.0f/(1.0f+std::exp(-x));
        case Act::Tanh:    return std::tanh(x);
    }
}
static float ActDeriv(Act a, float y){
    switch(a){
        default: 
        case Act::Linear: return 1.0f;
        case Act::ReLU:   return y>0.f?1.f:0.f;
        case Act::Sigmoid: return y*(1.0f-y);
        case Act::Tanh:    return 1.0f - y*y;
    }
}

// Persistent graph cache: ensure it is built only when topology changes
static void EnsureCache(NeuralGraph& g){
    if(g.cache.valid && g.cache.version == g.topoVersion) return;
    g.cache.id2idx.clear();
    g.cache.outEdges.clear();
    g.cache.inEdges.clear();
    for(size_t i=0;i<g.nodes.size();++i) g.cache.id2idx[g.nodes[i].id] = (int)i;
    for(size_t i=0;i<g.edges.size();++i){
        const auto& e=g.edges[i];
        g.cache.outEdges[e.a].push_back((int)i);
        g.cache.inEdges[e.b].push_back((int)i);
    }
    g.cache.version = g.topoVersion;
    g.cache.valid = true;
}

static void ForwardPass(NeuralGraph& g){
    EnsureCache(g);
    for(const auto& L : g.layers){
        if(L.kind == LayerKind::Input) continue;
        for(int idx : L.nodeIdx){
            Neuron& n = g.nodes[idx];
            float v_ = 0.0f;
            bool usedOverride = false;
            if(L.kind == LayerKind::Output && gUseManualOverrides && !gTrain.active && !g.liveInputs){
                auto itov = gManualOutputs.find(n.id);
                if(itov != gManualOutputs.end()){
                    v_ = itov->second; usedOverride = true;
                }
            }
            if(!usedOverride){
                float sum = ng_get_bias(n);
                auto it = g.cache.inEdges.find(n.id);
                if(it != g.cache.inEdges.end()){
                    for(int ei : it->second){
                        const auto& e = g.edges[ei];
                        auto ita = g.cache.id2idx.find(e.a);
                        if(ita!=g.cache.id2idx.end()) sum += e.w * g.nodes[ita->second].value;
                    }
                }
                v_ = ActApply(L.act, sum);
            }
            if(L.kind == LayerKind::Output){ if(v_ < 0.0f) v_ = 0.0f; else if(v_ > 1.0f) v_ = 1.0f; }
            n.value = v_;
        }
    }
}

static float sgn(float x){ return (x>0)-(x<0); }

static float TrainOneEpoch(NeuralGraph& g, float lr, float l1, float l2){
    if(gDataset.empty()) return FLT_MAX;
    float mse = 0.0f;

    EnsureCache(g);

    for(const auto& s : gDataset){
        size_t pi=0;
        for(const auto& L : g.layers){
            if(L.kind != LayerKind::Input) continue;
            for(int idx : L.nodeIdx){
                g.nodes[idx].value = (pi < s.in.size() ? s.in[pi++] : 0.0f);
            }
        }
        ForwardPass(g);

        std::vector<int> outNodeIdx;
        Act outAct = Act::Linear;
        for(const auto& L : g.layers){
            if(L.kind == LayerKind::Output){
                outAct = L.act;
                for(int idx : L.nodeIdx) outNodeIdx.push_back(idx);
            }
        }

        std::vector<float> delta(g.nodes.size(), 0.0f);
        for(size_t i=0; i<outNodeIdx.size(); ++i){
            int idx = outNodeIdx[i];
            Neuron& n = g.nodes[idx];
            float y = n.value;
            float t = (i < s.out.size() ? s.out[i] : 0.0f);
            float err = y - t;
            mse += err*err;
            float d = err * ActDeriv(outAct, y);
            delta[idx] = d;
        }

        // Hidden layers backprop (right-to-left; skip last layer already handled)
        for(int Li=(int)g.layers.size()-2; Li>=0; --Li){
            const auto& L = g.layers[Li];
            if(L.kind == LayerKind::Input) continue;
            Act act = L.act;
            for(int idx : L.nodeIdx){
                Neuron& n = g.nodes[idx];
                float sum=0.0f;
                auto it = g.cache.outEdges.find(n.id);
                if(it != g.cache.outEdges.end()){
                    for(int ei : it->second){
                        const auto& e = g.edges[ei];
                        auto itb = g.cache.id2idx.find(e.b);
                        if(itb != g.cache.id2idx.end()) sum += e.w * delta[itb->second];
                    }
                }
                float d = ActDeriv(act, n.value) * sum;
                delta[idx] = d;
            }
        }

        // Weights update
        for(auto& e : g.edges){
            float srcv = 0.0f;
            auto ita = g.cache.id2idx.find(e.a);
            if(ita!=g.cache.id2idx.end()) srcv = g.nodes[ita->second].value;
            float db = 0.0f; auto itb = g.cache.id2idx.find(e.b); if(itb!=g.cache.id2idx.end()) db = delta[itb->second];
            float grad = srcv * db + l2*e.w + l1*sgn(e.w);
            e.w -= lr * grad;
        }
        // Bias update
        for(const auto& L : g.layers){
            if(L.kind == LayerKind::Input) continue;
            for(int idx : L.nodeIdx){
                Neuron& n = g.nodes[idx];
                float db = delta[idx];
                float b = ng_get_bias(n);
                ng_set_bias(n, b - lr * db);
            }
        }
    }
    mse /= (float)std::max<size_t>(1, gDataset.size());
    if(gEngine) gEngine->networkDirty = true;
    return mse;
}

#endif

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Neural Editor implementation
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#ifdef USE_IMGUI
[[maybe_unused]] static ImU32 colU32(float r,float g,float b,float a=1.f){ return ImGui::GetColorU32(ImVec4(r,g,b,a)); }

// Forward declaration of global engine pointer for theme access
extern Engine* gEngine;

Neuron* NeuralGraph::findAt(float mx, float my){
    for(auto& n: nodes){
        float dx=mx-n.x, dy=my-n.y;
        if(dx*dx+dy*dy <= n.radius*n.radius) return &n;
    }
    return nullptr;
}

void NeuralGraph::buildDemo(){
    nodes.clear(); edges.clear(); layers.clear();
    selected = -1; std::memset(editBuf,0,sizeof(editBuf));
    currentLayer = 0;
    liveInputs = false;

    int nIn = 3;
    std::array<int,3> nHL = {5, 8, 5};
    int nOut = 3;

    int id = 0;

    // Inputs
    {
        Layer L; 
        L.kind = LayerKind::Input; 
        L.name = "Inputs"; 
        L.act = Act::Linear;
        layers.push_back(L);
        for(int i=0;i<nIn;i++){
            Neuron n; n.id=id++; n.isInput=true; n.label = "In"+std::to_string(i+1); n.speed = 0.0f; n.amp = 1.0f;
            nodes.push_back(n); layers.back().nodeIdx.push_back((int)nodes.size()-1);
        }
    }

    // HL1..HL3
    for(int li=0; li<3; ++li){
        Layer L; 
        L.kind = LayerKind::Hidden; 
        L.name = "HL " + std::to_string(li+1);
        L.act = Act::ReLU;
        for(int i=0;i<nHL[li]; ++i){
            Neuron n; n.id=id++; n.isInput=false; n.label = "H"+std::to_string(li+1)+"."+std::to_string(i+1);
            nodes.push_back(n); L.nodeIdx.push_back((int)nodes.size()-1);
        }
        layers.push_back(std::move(L));
    }

    // Outputs
    {
        Layer L; L.kind = LayerKind::Output; 
        L.name = "Outputs"; 
        L.act = Act::Sigmoid;
        layers.push_back(L);
        for(int i=0;i<nOut;i++){
            Neuron n; n.id=id++; n.isInput=false; n.label = "Out"+std::to_string(i+1);
            nodes.push_back(n); layers.back().nodeIdx.push_back((int)nodes.size()-1);
        }
    }

    NG_RebuildEdges(*this);
    layoutLayers(800.0f, 500.0f);
}

void NeuralGraph::layoutLayers(float W, float H){
    const float marginX = 40.f, marginY = 40.f;
    const float colGap  = (W - 2*marginX);
    const float colW    = colGap / std::max<size_t>(1, layers.size());

    // Compute a global radius so that all neurons fit vertically and horizontally
    float headerH = 3.0f * ImGui::GetTextLineHeightWithSpacing() + 6.0f; // 3 header lines
    float y0g = marginY + headerH;
    float y1g = H - marginY;
    float minSpacingY = 1e9f;
    for(const auto& Lchk : layers){
        int N = (int)Lchk.nodeIdx.size();
        if(N>1){
            float span = (y1g - y0g);
            float sp = span / (float)(N-1);
            if(sp < minSpacingY) minSpacingY = sp;
        }
    }
    if(minSpacingY == 1e9f) minSpacingY = (y1g - y0g); // single-node layers
    float rVert = std::max(6.0f, 0.5f*minSpacingY - 6.0f);
    // Horizontal limit: ensure circle fits within column; leave padding
    float rHorizGeneric = std::max(6.0f, 0.5f*colW - 12.0f);
    // Inputs have a mini plot on the left: ensure it stays within canvas
    const float kMiniPlotW = 48.0f;
    float rHorizInput = std::max(6.0f, 0.5f*colW - (10.0f + kMiniPlotW + 6.0f));
    float rHoriz = std::min(rHorizGeneric, rHorizInput);
    float globalR = std::clamp(std::min(rVert, rHoriz), 6.0f, 38.0f);

    float x = marginX;
    for(size_t li=0; li<layers.size(); ++li){
        Layer& L = layers[li];
        int N = (int)L.nodeIdx.size();
        float y0 = y0g;
        float y1 = y1g;
        float span = (N>1) ? (y1 - y0) : 0.f;
        for(int i=0;i<N;i++){
            int idx = L.nodeIdx[i];
            float yy = (N>1)? y0 + (span * (float)i/(float)(N-1)) : (W*0.5f);
            nodes[idx].x = x + colW*0.5f;
            nodes[idx].y = yy;
            nodes[idx].radius = globalR;
        }
        L.x0 = x + 6.f;
        L.x1 = x + colW - 6.f;
        L.y0 = marginY * 0.5f;
        L.y1 = H - marginY * 0.3f;
        x += colW;
    }
}

void NeuralGraph::tickLiveInputs(float dt){
    // Per-input live animation; global flag enables all inputs if true
    if(!(liveInputs)){
        // still allow per-neuron live without global toggle
    }
    // Do not clear manual output overrides here; they are used when liveInputs is off
    if(layers.empty()) return;
    
    // Update live feed wave effect
    if(gLiveFeedWaveActive){
        gLiveFeedWavePhase += gLiveFeedWaveSpeed * dt;
        gLiveFeedInputTimer += dt;
        
        // Check if it's time to activate the next input
        if(gLiveFeedInputTimer >= gLiveFeedInputDelay){
            gLiveFeedInputTimer = 0.0f;
            gLiveFeedCurrentInput++;
            
            // Reset to first input if we've gone through all
            if(gLiveFeedCurrentInput >= (int)layers.front().nodeIdx.size()){
                gLiveFeedCurrentInput = 0;
            }
        }
    }
    
    for(int i = 0; i < (int)layers.front().nodeIdx.size(); ++i){
        int idx = layers.front().nodeIdx[i];
        Neuron& n = nodes[idx];
        if(!n.isInput) continue;
        
        // Handle wave effect
        if(gLiveFeedWaveActive){
            if(i == gLiveFeedCurrentInput){
                // Current input in wave - activate with sine wave
                n.value = 0.5f + 0.5f * std::sin(gLiveFeedWavePhase);
            } else {
                // Other inputs - gradually fade to 0
                n.value = std::max(0.0f, n.value - dt * 2.0f);
            }
        } else if(liveInputs || n.live){
            // Normal live input behavior
            n.phase += 2.0f * 3.1415926535f * std::max(0.0f, n.speed) * dt;
            float a = std::clamp(n.amp, 0.0f, 1.0f);
            n.value = 0.5f + 0.5f * a * std::sin(n.phase);
        }
    }
}

void NeuralGraph::stepLayout(float dt, float W, float H){
    tickLiveInputs(dt);

    
    // Recompute forward pass whenever inputs may have changed
    ForwardPass(*this);

const float rep = 1200.0f;
    const float damp= 0.90f;
    for(size_t idxA=0; idxA<nodes.size(); ++idxA){
        auto& a = nodes[idxA];
        float fx=0, fy=0;
        for(size_t idxB=0; idxB<nodes.size(); ++idxB){
            if(idxA==idxB) continue;
            auto& b = nodes[idxB];
            float dx=a.x-b.x, dy=a.y-b.y;
            float d2 = dx*dx+dy*dy + 1e-2f;
            float inv = 1.0f / d2;
            float f = rep * inv;
            fx += dx * f; fy += dy * f;
        }
        float targetX = a.x;
        for(auto& L: layers){
            if(std::find(L.nodeIdx.begin(), L.nodeIdx.end(), (int)idxA) != L.nodeIdx.end()){
                targetX = (L.x0 + L.x1)*0.5f;
                break;
            }
        }
        fx += (targetX - a.x) * 5.0f;
        fy += (H*0.5f - a.y) * 0.4f;

        a.vx = (a.vx + fx*dt) * damp;
        a.vy = (a.vy + fy*dt) * damp;
    }
    for(auto& a: nodes){
        a.x += a.vx*dt; a.y += a.vy*dt;
        a.x = std::clamp(a.x, 30.0f, W-30.0f);
        a.y = std::clamp(a.y, 30.0f, H-30.0f);
    }
}

void NeuralGraph::drawCanvas(bool& open, float width, float height){
    if(!open) return;

    extern bool gForceRelayoutWindows;
    ImGuiCond cond = gForceRelayoutWindows ? ImGuiCond_Always : ImGuiCond_FirstUseEver;
    ImGuiWindowFlags winFlags = ImGuiWindowFlags_NoCollapse;
    if(gFullscreenPreview){
        ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->Pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(vp->Size, ImGuiCond_Always);
        winFlags |= ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
    } else {
        ImGui::SetNextWindowSize(ImVec2(width, height), cond);
        ImGui::SetNextWindowPos(ImVec2(10,10), cond);
    }

    char editorTitle[128];
    std::snprintf(editorTitle, sizeof(editorTitle), "Neural Network [F1] (Neurons: %zu)###Neural Editor", nodes.size());
    if(ImGui::Begin(editorTitle, &open, winFlags)){
        // Block shortcuts while editing labels; reset per frame and set when inputs are focused
        gEditingLabel = false;
        // split: top=canvas, bottom=live feed
        ImVec2 avail = ImGui::GetContentRegionAvail();
        if(avail.x < 100) avail.x = 100;
        if(avail.y < 140) avail.y = 140;

        float canvasH = avail.y; ImVec2 canvasSize = ImVec2(avail.x, canvasH);

        // Auto relayout when size changes
        if(std::abs(canvasSize.x - lastCanvasW) > 1.0f || std::abs(canvasSize.y - lastCanvasH) > 1.0f){
            layoutLayers(canvasSize.x, canvasSize.y);
            lastCanvasW = canvasSize.x; lastCanvasH = canvasSize.y;
        }

        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 origin = ImGui::GetCursorScreenPos();
        ImVec2 canvasMax = ImVec2(origin.x + canvasSize.x, origin.y + canvasSize.y);

        // CANVAS background
        dl->AddRectFilled(origin, canvasMax, ImGui::GetColorU32(ImVec4(0.08f,0.08f,0.095f,1.0f)), 6.0f);
        dl->AddRect(origin, canvasMax, ImGui::GetColorU32(ImVec4(0.25f,0.25f,0.3f,1.0f)), 6.0f);

        ImVec2 _csize = ImVec2(std::max(1.0f, canvasSize.x), std::max(1.0f, canvasSize.y));
        ImGui::InvisibleButton("canvas", _csize);
        bool isHovered = ImGui::IsItemHovered();
        ImVec2 mouse = ImGui::GetIO().MousePos;
        float mx = mouse.x - origin.x, my = mouse.y - origin.y;

        // layout tick + animation
        float dt = ImGui::GetIO().DeltaTime;
        stepLayout(std::max(0.0f, dt), canvasSize.x, canvasSize.y);

        // layer backgrounds + multiâ€‘line headers + click activation
        for(size_t li=0; li<layers.size(); ++li){
            Layer& L = layers[li];
            ImVec2 a(origin.x + L.x0, origin.y + L.y0);
            ImVec2 b(origin.x + L.x1, origin.y + L.y1);
            ImU32 colBg = (int(li)==currentLayer) ?
                ImGui::GetColorU32(ImVec4(0.18f,0.20f,0.28f,0.95f)) :
                ImGui::GetColorU32(ImVec4(0.12f,0.12f,0.15f,0.75f));
            dl->AddRectFilled(a, b, colBg, 8.0f);
            dl->AddRect(a, b, ImGui::GetColorU32(ImVec4(0.35f,0.35f,0.40f,0.9f)), 8.0f, 0, 2.0f);

            // --- 3 header lines:
            float lh = ImGui::GetTextLineHeightWithSpacing();
            ImVec2 hp(a.x + 8.0f, a.y + 6.0f);

            // 1) layer name
            dl->AddText(hp, ImGui::GetColorU32(ImVec4(1,1,0.85f,1)), L.name.c_str());
            // 2) neuron count
            char ln[64]; std::snprintf(ln, sizeof(ln), "Neurons: %d", (int)L.nodeIdx.size());
            dl->AddText(ImVec2(hp.x, hp.y + lh), ImGui::GetColorU32(ImVec4(0.85f,0.9f,1,1)), ln);
            // 3) activation - clickable
            const char* actName = ActName(L.act);
            char la[96]; std::snprintf(la, sizeof(la), "Activation: %s  (click)", actName);
            ImVec2 actPos(hp.x, hp.y + lh*2.0f);
            dl->AddText(actPos, ImGui::GetColorU32(ImVec4(0.9f,0.8f,1,1)), la);

            ImVec2 actMin = actPos;
            ImVec2 actMax = ImVec2(b.x - 8.0f, actPos.y + lh);
            if(ImGui::IsMouseHoveringRect(actMin, actMax) &&
               ImGui::IsMouseClicked(ImGuiMouseButton_Left)){
                L.act = NextAct(L.act);
            }

            // click entire column = select layer (LPM/PPM)
            ImGui::SetCursorScreenPos(a);
            ImGui::InvisibleButton(("layerbtn##"+std::to_string(li)).c_str(),
                                   ImVec2(std::max(1.0f, b.x-a.x), std::max(1.0f, b.y-a.y)));
            if(ImGui::IsItemClicked(ImGuiMouseButton_Left) || ImGui::IsItemClicked(ImGuiMouseButton_Right)){
                currentLayer = (int)li;
            }
        }

        // compute top output neuron (for highlight)
        int topOutId = -1; float topOutVal = -FLT_MAX;
        for(const auto& L_ : layers){
            if(L_.kind != LayerKind::Output) continue;
            for(int idx_ : L_.nodeIdx){
                if(idx_>=0 && idx_<(int)nodes.size()){
                    const Neuron& nn_ = nodes[idx_];
                    if(nn_.value > topOutVal){ topOutVal = nn_.value; topOutId = nn_.id; }
                }
            }
        }

        // compute top input neuron (for label background)
        int topInId = -1; float topInVal = -FLT_MAX;
        for(const auto& L_ : layers){
            if(L_.kind != LayerKind::Input) continue;
            for(int idx_ : L_.nodeIdx){
                if(idx_>=0 && idx_<(int)nodes.size()){
                    const Neuron& nn_ = nodes[idx_];
                    if(nn_.value > topInVal){ topInVal = nn_.value; topInId = nn_.id; }
                }
            }
        }

        // edges
        auto findById = [&](int id)->Neuron*{
            for(auto& n: nodes){
                if(n.id==id) return &n;
            }
            return nullptr;
        };
        const UITheme& _theme = (gEngine ? gEngine->renderer.theme : UITheme::Cyberpunk());
        for(auto& e: edges){
            Neuron* A = findById(e.a);
            Neuron* B = findById(e.b);
            if(!A||!B) continue;
            ImVec2 p1(origin.x+A->x, origin.y+A->y), p2(origin.x+B->x, origin.y+B->y);
            // (thickness computed below)
            // Map weight [-1,1] -> theme gradient and style
            {
                float t = (e.w + 1.0f) * 0.5f; t = t<0?0: (t>1?1:t);
                auto mixf = [](float a, float b, float tt){ return a + (b-a)*tt; };
                float r = mixf(_theme.edgeStart.x, _theme.edgeEnd.x, t);
                float g = mixf(_theme.edgeStart.y, _theme.edgeEnd.y, t);
                float b = mixf(_theme.edgeStart.z, _theme.edgeEnd.z, t);
                float a = _theme.edgeAlpha;
                float thick = _theme.edgeBase + _theme.edgeScale * std::fabs(e.w);
                // Pulsing glow
                float time = (float)ImGui::GetTime();
                float pulse = 0.5f + 0.5f*std::sin(time*2.0f + 0.15f*(float)(A->id + B->id));
                ImU32 glowCol = ImGui::GetColorU32(ImVec4(r,g,b, 0.18f * pulse));
                float w1 = thick * (2.5f + 2.0f*pulse);
                float w2 = thick * (1.8f + 1.2f*pulse);
                float w3 = thick * (1.2f + 0.6f*pulse);
                // Use actual panel values or oscillating values based on sine wave settings
                float currentStrength = gStrengthSineEnabled ? 
                    (150.0f + 150.0f * std::sin(gWindStrengthPhase)) : 
                    gWind.strength;
                float currentSpeed = gSpeedSineEnabled ? 
                    (2.5f + 2.5f * std::sin(gWindSpeedPhase)) : 
                    gWind.speed;
                float currentJitter = gJitterSineEnabled ? 
                    (0.25f + 0.25f * std::sin(gWindJitterPhase)) : 
                    gWind.jitter;

                // Wind sway: stronger turbine => more bend, same direction for all (with slight jitter)
                float rpm = 0.0f;
                if(gEngine){
                    rpm = gEngine->turbine.params.rpm;
                }
                // Normalize RPM to [0,1] using a soft scale; tweakable mapping
                float windK = std::fmin(rpm / 2000.0f, 1.0f);
                // Base amplitude in pixels, smoothly interpolated
                float baseAmp = (currentStrength * windK) * gWindInterpolant;
                // Global wind direction (mostly to the right, slightly up). Add tiny rotation jitter.
                float baseDirX = 1.0f, baseDirY = -0.15f;
                float jitter = currentJitter * std::sin(time*0.8f + 0.07f*(A->id*13 + B->id*17));
                float ca = std::cos(jitter), sa = std::sin(jitter);
                ImVec2 windDir = ImVec2(baseDirX*ca - baseDirY*sa, baseDirX*sa + baseDirY*ca);
                // Mild oscillation so they sway a bit over time (same sense for all)
                float osc = 0.75f + 0.25f*std::sin(time * currentSpeed + 0.05f*(A->id + B->id));
                float amp = baseAmp * osc;
                ImVec2 wind = ImVec2(windDir.x * amp, windDir.y * amp);

                // BĂ©zier control points pushed along wind to create a smooth bend
                ImVec2 d = ImVec2(p2.x - p1.x, p2.y - p1.y);
                ImVec2 c1 = ImVec2(p1.x + d.x*0.33f + wind.x, p1.y + d.y*0.33f + wind.y);
                ImVec2 c2 = ImVec2(p1.x + d.x*0.66f + wind.x, p1.y + d.y*0.66f + wind.y);

                dl->AddBezierCubic(p1, c1, c2, p2, glowCol, w1);
                dl->AddBezierCubic(p1, c1, c2, p2, glowCol, w2);
                dl->AddBezierCubic(p1, c1, c2, p2, glowCol, w3);
                dl->AddBezierCubic(p1, c1, c2, p2, ImGui::GetColorU32(ImVec4(r,g,b,a)), thick);
            }
        }

        // selection handled below; mini-plot is not clickable

        // neurons + labels + mini-sine + speed slider
        for(auto& n: nodes){
            ImVec2 p(origin.x+n.x, origin.y+n.y);
            bool isSel = (selected==n.id);

            // compute if this neuron is in the output layer
            bool isOutput = false;
            {
                int idxN = int(&n - &nodes[0]);
                for(const auto& L: layers){
                    if(L.kind != LayerKind::Output) continue;
                    for(int j : L.nodeIdx){ if(j==idxN){ isOutput = true; break; } }
                    if(isOutput) break;
                }
            }

            auto colU = [&](const Vec3& v){ return ImGui::GetColorU32(ImVec4(v.x,v.y,v.z,1.0f)); };
            ImU32 baseFill = n.isInput ? colU(_theme.nInput) : (isOutput ? colU(_theme.nOutput) : colU(_theme.nHidden));
            ImU32 fill = isSel ? ImGui::GetColorU32(ImVec4(0.2f,0.5f,1.0f,1.0f)) : baseFill;
            ImU32 ring = isSel ? colU(_theme.nRing)
                               : ImGui::GetColorU32(ImVec4(0.9f,0.9f,0.95f,0.9f));

            // highlight top output: theme highlight
            if(isOutput && n.id == topOutId){
                fill = colU(_theme.nHighlight);
            }

            dl->AddCircleFilled(p, n.radius, fill, 32);
            dl->AddCircle(p, n.radius, ring, 32, 2.0f);

            // Determine if this neuron is in a hidden layer
            bool isHidden = false;
            {
                int idxN = int(&n - &nodes[0]);
                for(const auto& L: layers){
                    if(L.kind != LayerKind::Hidden) continue;
                    for(int j : L.nodeIdx){ if(j==idxN){ isHidden = true; break; } }
                    if(isHidden) break;
                }
            }

            // label: hidden BELOW, input on LEFT (leave space for mini-plot), output on RIGHT
            ImVec2 ts = ImGui::CalcTextSize(n.label.c_str());
            ImVec2 tl;
            if(isHidden){
                tl = ImVec2(p.x - ts.x*0.5f, p.y + n.radius + 4.0f);
            } else if(isOutput){
                tl = ImVec2(p.x + n.radius + 8.0f, p.y - ts.y*0.5f);
            } else {
                tl = ImVec2(p.x - n.radius - 8.0f - ts.x - 56.0f, p.y - ts.y*0.5f); // input - space for mini-plot on the left
            }

            // Yellow label background for top input/output by value
            bool isTopIn  = (!isHidden && n.isInput && n.id==topInId);
            bool isTopOut = (isOutput && n.id==topOutId);
            if(isTopIn || isTopOut){
                ImVec2 pad(4.0f, 2.0f);
                ImVec2 r0 = ImVec2(tl.x - pad.x, tl.y - pad.y);
                ImVec2 r1 = ImVec2(tl.x + ts.x + pad.x, tl.y + ts.y + pad.y);
                ImU32 bg = ImGui::GetColorU32(ImVec4(1.0f,0.95f,0.30f,0.90f));
                ImU32 bd = ImGui::GetColorU32(ImVec4(0.85f,0.75f,0.10f,0.95f));
                dl->AddRectFilled(r0, r1, bg, 3.0f);
                dl->AddRect(r0, r1, bd, 3.0f, 0, 1.0f);
            }
            {
                ImU32 labelCol = ImGui::GetColorU32( (isTopIn || isTopOut)
                    ? ImVec4(0.0f,0.0f,0.0f,1.0f)
                    : ImVec4(1.0f,1.0f,0.85f,1.0f) );
                dl->AddText(tl, labelCol, n.label.c_str());
                // Tooltip over label area
                ImVec2 br = ImVec2(tl.x + ts.x, tl.y + ts.y);
                if(ImGui::IsMouseHoveringRect(tl, br)){
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(n.desc.empty()? "default description" : n.desc.c_str());
                    ImGui::EndTooltip();
                }
            }

            // Manual override icon next to output labels when active
            if(isOutput){
                bool manualActive = false;
                if(gUseManualOverrides && !gTrain.active && !liveInputs){
                    manualActive = (gManualOutputs.find(n.id) != gManualOutputs.end());
                }
                if(manualActive){
                    ImVec2 iconC = ImVec2(tl.x - 10.0f, p.y);
                    ImU32 icoCol = ImGui::GetColorU32(ImVec4(1.0f,0.95f,0.3f,1.0f)); // yellow icon
                    dl->AddCircleFilled(iconC, 4.0f, icoCol, 16);
                    // Tooltip area
                    ImGui::SetCursorScreenPos(ImVec2(iconC.x - 6.0f, iconC.y - 6.0f));
                    ImGui::InvisibleButton((std::string("ovr##")+std::to_string(n.id)).c_str(), ImVec2(12,12));
                    if(ImGui::IsItemHovered()) ImGui::SetTooltip("Manual override active");
                }
            }

            // show value inside output neurons too
            if(isOutput){
                char bufv[16]; std::snprintf(bufv,sizeof(bufv),"%.2f", n.value);
                ImVec2 tsv2 = ImGui::CalcTextSize(bufv);
                dl->AddText(ImVec2(p.x - tsv2.x*0.5f, p.y - tsv2.y*0.5f),
                            ImGui::GetColorU32(ImVec4(0.05f,0.05f,0.08f,1.0f)), bufv);
            }

            if(n.isInput){
                char buf[16]; std::snprintf(buf,sizeof(buf),"%.2f", n.value);
                ImVec2 tsv = ImGui::CalcTextSize(buf);
                dl->AddText(ImVec2(p.x - tsv.x*0.5f, p.y - tsv.y*0.5f),
                            ImGui::GetColorU32(ImVec4(0.05f,0.05f,0.08f,1.0f)), buf);

                // Mini live-feed plot moved to LEFT of neuron
                const float PW=48.f, PH=22.f;
                ImVec2 pp0(p.x - n.radius - 10.f - PW, p.y - PH*0.5f);
                ImVec2 pp1(pp0.x + PW, pp0.y + PH);
                ImU32 plotBg = ImGui::GetColorU32(ImVec4(0.1f,0.12f,0.14f,1.0f));
                ImU32 plotFg = ImGui::GetColorU32(ImVec4(0.2f,0.85f,0.7f,1.0f));
                if(n.live){ plotBg = ImGui::GetColorU32(ImVec4(0.12f,0.16f,0.12f,1.0f)); plotFg = ImGui::GetColorU32(ImVec4(0.3f,0.9f,0.3f,1.0f)); }
                dl->AddRectFilled(pp0, pp1, plotBg, 3.0f);
                dl->AddRect(pp0, pp1, ImGui::GetColorU32(ImVec4(0.35f,0.35f,0.40f,1.0f)), 3.0f, 0, 1.0f);
                const int S=32;
                ImVec2 prev = {pp0.x, pp0.y + PH*(1.0f - (0.5f+0.5f*std::sin(n.phase - 2*3.14159f)))};
                for(int i=1;i<=S;i++){
                    float t=(float)i/(float)S;
                    float val = 0.5f + 0.5f*std::sin(n.phase + t*2.0f*3.14159265f);
                    ImVec2 cur = { pp0.x + t*PW, pp0.y + PH*(1.0f - val) };
                    dl->AddLine(prev, cur, plotFg, 1.5f);
                    prev = cur;
                }
                // mini plot is not clickable
            }
        }

        // click elsewhere on canvas selects neuron but doesn't toggle live/value
        if(isHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)){
            if(Neuron* hit = findAt(mx,my)){
                // Allow selection only on inputs and outputs
                bool isOutput = false;
                for(const auto& Lchk : layers){
                    if(Lchk.kind == LayerKind::Output){
                        for(int idxN : Lchk.nodeIdx){ if(nodes[idxN].id == hit->id){ isOutput = true; break; } }
                    }
                    if(isOutput) break;
                }
                if(hit->isInput || isOutput){
                    ImGuiIO& io = ImGui::GetIO();
                    if(io.KeyShift){ hit->value = 1.0f; gManualOutputs[hit->id] = 1.0f; gUseManualOverrides = true; }
                    if(io.KeyCtrl){ hit->value = 0.0f; gManualOutputs[hit->id] = 0.0f; gUseManualOverrides = true; }
                    selected = hit->id;
                    std::snprintf(editBuf, sizeof(editBuf), "%s", hit->label.c_str());
                    ForwardPass(*this);
                }
            } else { selected = -1; std::memset(editBuf,0,sizeof(editBuf)); }
        }

        // Properties panel: child window at bottom-left for selected neuron
        if(selected != -1){
            Neuron* sel = nullptr;
            for(auto& n_ : nodes){ if(n_.id==selected){ sel=&n_; break; } }
            if(sel){
                bool isOutput = false;
                for(const auto& Lchk : layers){
                    if(Lchk.kind == LayerKind::Output){
                        for(int idxN : Lchk.nodeIdx){ if(nodes[idxN].id == sel->id){ isOutput = true; break; } }
                    }
                    if(isOutput) break;
                }
                const float panelW=360.0f, panelH=168.0f;
                ImVec2 pos = ImVec2(origin.x + 10.0f, origin.y + canvasSize.y - panelH - 10.0f);
                ImGui::SetCursorScreenPos(pos);
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.05f,0.05f,0.07f,0.95f));
                ImGui::BeginChild("nn_props", ImVec2(panelW, panelH), true, ImGuiWindowFlags_NoScrollbar);
                ImGui::Text("Neuron ID %d", sel->id);
                ImGui::PushItemWidth(220.0f);
                bool labelEnter = ImGui::InputText("Label", editBuf, IM_ARRAYSIZE(editBuf), ImGuiInputTextFlags_EnterReturnsTrue);
                static float pendingVal=0.0f, pendingSpeed=0.0f; static int lastSel=-1;
                if(lastSel!=sel->id){ pendingVal=sel->value; pendingSpeed=sel->speed; lastSel=sel->id; }
                if(isOutput){
                    ImGui::SliderFloat("Value", &pendingVal, 0.0f, 1.0f, "%.3f");
                } else if(sel->isInput){
                    ImGui::SliderFloat("Value", &pendingVal, 0.0f, 1.0f, "%.3f");
                    ImGui::SliderFloat("Speed (Hz)", &pendingSpeed, 0.0f, 1.0f, "%.3f");
                    ImGui::Checkbox("Live", &sel->live);
                } else {
                    ImGui::BeginDisabled(); float vtmp=sel->value; ImGui::SliderFloat("Value", &vtmp, -10.0f, 10.0f, "%.3f"); ImGui::EndDisabled();
                }
                if(ImGui::Button("Apply") || labelEnter){
                    sel->label = editBuf;
                    if(isOutput){
                        pendingVal = std::clamp(pendingVal, 0.0f, 1.0f);
                        sel->value = pendingVal; gManualOutputs[sel->id] = pendingVal; gUseManualOverrides = true;
                    } else if(sel->isInput){
                        pendingVal = std::clamp(pendingVal, 0.0f, 1.0f);
                        sel->value = pendingVal; sel->speed = std::max(0.0f, pendingSpeed);
                        liveInputs = false;
                    }
                    ForwardPass(*this);
                }
                ImGui::PopItemWidth();
                ImGui::EndChild();
                ImGui::PopStyleColor();
            }
        }

        // ====== LIVE FEED moved to separate window (F6); disabled here ======
        if(gHUDHasRect){
        ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + canvasSize.y + 8.0f));
        ImGui::BeginChild("live_feed", ImVec2(avail.x, 0), true, ImGuiWindowFlags_HorizontalScrollbar);
        {
            const Layer& L = layers[std::clamp(currentLayer, 0, (int)layers.size()-1)];
            ImGui::Text("Live Feed - %s (neurons: %d, activation: %s)",
                        L.name.c_str(), (int)L.nodeIdx.size(), ActName(L.act));
            if(ImGui::BeginTable("lf", 6, ImGuiTableFlags_Resizable|ImGuiTableFlags_RowBg|ImGuiTableFlags_BordersInnerV)){
                ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 40.f);
                ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 80.f);
                ImGui::TableSetupColumn("Speed(Hz)", ImGuiTableColumnFlags_WidthFixed, 90.f);
                ImGui::TableSetupColumn("In", ImGuiTableColumnFlags_WidthFixed, 40.f);
                ImGui::TableSetupColumn("Out", ImGuiTableColumnFlags_WidthFixed, 45.f);
                ImGui::TableHeadersRow();

                for(int idx : L.nodeIdx){
                    Neuron& n = nodes[idx];
                    int inCnt=0, outCnt=0;
                    for(auto& e: edges){ if(e.b==n.id) ++inCnt; if(e.a==n.id) ++outCnt; }

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::Text("%d", n.id);
                    ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(n.label.c_str());

                    ImGui::TableSetColumnIndex(2);
                    float v = n.value;
                    ImVec4 c = ImVec4(1.0f - v, 0.2f + 0.8f*v, 0.15f, 1.0f);
                    ImGui::TextColored(c, "%.3f", v);

                    ImGui::TableSetColumnIndex(3);
                    if(n.isInput){
                        ImGui::Text("%.2f", n.speed);
                    } else {
                        ImGui::TextDisabled("-");
                    }

                    ImGui::TableSetColumnIndex(4); ImGui::Text("%d", inCnt);
                    ImGui::TableSetColumnIndex(5); ImGui::Text("%d", outCnt);
                }
                ImGui::EndTable();
            }

            // Selected neuron editor
            if(selected != -1){
                Neuron* sel = nullptr;
                for(auto& n : nodes) if(n.id==selected){ sel=&n; break; }
                if(sel){
                    ImGui::Separator();
                    ImGui::Text(u8"Neuron editor (ID %d)", sel->id);
                    ImGui::PushItemWidth(320.f);

                    ImGui::InputText(u8"Label", editBuf, IM_ARRAYSIZE(editBuf));
                    bool labelEnter = false; // Enter handled globally below
                    gEditingLabel = ImGui::IsItemActive() || ImGui::IsItemFocused();

                    // Is output?
                    bool isOutput = false;
                    for(const auto& Lchk : layers){
                        if(Lchk.kind == LayerKind::Output){
                            for(int idxN : Lchk.nodeIdx){
                                if(nodes[idxN].id == sel->id){ isOutput = true; break; }
                            }
                        }
                        if(isOutput) break;
                    }

                    static int lastSelForVal = -1;
                    static float pendingVal = 0.0f;
                    static float pendingSpeed = 0.0f;
                    if(lastSelForVal != sel->id){
                        pendingVal = sel->value;
                        pendingSpeed = sel->speed;
                        std::snprintf(editBuf, sizeof(editBuf), "%s", sel->label.c_str());
                        lastSelForVal = sel->id;
                    }

                    // Description: reset input when switching selected neuron
                    static char descBuf[256];
                    static int lastSelForDesc = -1;
                    if(lastSelForDesc != sel->id){
                        // Clear the description field on neuron change (per request)
                        descBuf[0] = '\0';
                        lastSelForDesc = sel->id;
                    }
                    ImGui::InputTextMultiline("Description", descBuf, IM_ARRAYSIZE(descBuf), ImVec2(320, 60));
                    gEditingLabel = gEditingLabel || ImGui::IsItemActive() || ImGui::IsItemFocused();

                    if(isOutput){
                        ImGui::SliderFloat(u8"Value", &pendingVal, 0.0f, 1.0f, "%.4f");
                    } else if(sel->isInput){
                        ImGui::SliderFloat(u8"Value", &pendingVal, 0.0f, 1.0f, "%.4f");
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(120.f);
                        ImGui::InputFloat("##val", &pendingVal, 0.0f, 0.0f, "%.4f");
                    }
else{
                        ImGui::BeginDisabled();
                        float vtmp = sel->value;
                        ImGui::SliderFloat(u8"Value", &vtmp, -10.0f, 10.0f, "%.4f");
                        ImGui::EndDisabled();
                    }

                    if(sel->isInput){
                        if(ImGui::SliderFloat(u8"Speed (Hz)", &pendingSpeed, 0.0f, 1.0f, "%.3f")){
                            // Quantize to 0.001 increments
                            pendingSpeed = std::round(pendingSpeed * 1000.0f) / 1000.0f;
                            float prev = sel->speed;
                            sel->speed = std::max(0.0f, pendingSpeed);
                            // If user sets speed > 0, enable per-neuron live and reset phase for visible motion
                            if(sel->speed > 0.0f) {
                                if(prev <= 0.0f) sel->phase = 0.0f;
                                sel->live = true;
                            } else {
                                // speed == 0 stops motion; keep value as-is; live flag can remain
                            }
                        }
                        ImGui::Checkbox(u8"Live feed", &sel->live);
                    } else {
                        ImGui::BeginDisabled(); float dummy=0.f;
                        ImGui::SliderFloat(u8"Speed (Hz)", &dummy, 0.0f, 0.0f, "-");
                        ImGui::EndDisabled();
                    }

                    bool doApply = false;
                    doApply |= ImGui::Button(u8"Apply");
                    // Enter anywhere in this window applies changes
                    bool enterPressed = ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter);
                    doApply |= labelEnter || enterPressed;
                    if(doApply){
                        // Commit label
                        sel->label = editBuf;
                        sel->desc = descBuf;
                        // Commit value behavior
                        if(isOutput){
                            if(pendingVal < 0.0f) pendingVal = 0.0f; else if(pendingVal > 1.0f) pendingVal = 1.0f;
                            sel->value = pendingVal;
                            gManualOutputs[sel->id] = pendingVal;
                            gUseManualOverrides = true;
                        } else if(sel->isInput){
                            if(pendingVal < 0.0f) pendingVal = 0.0f; else if(pendingVal > 1.0f) pendingVal = 1.0f;
                            sel->value = pendingVal;
                            sel->speed = std::max(0.0f, pendingSpeed);
                            liveInputs = false;
                        }
                        ForwardPass(*this);
                    }

                    ImGui::PopItemWidth();
                }
            }
        }
        ImGui::EndChild();
        }
        // ImGui::EndChild();
        if(false) ImGui::EndChild();

        // Local actions removed — Neural Network shows only the network visualization.
    }
    ImGui::End();
}
#endif

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Engine (GLFW glue + loop)
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Engine* gEngine = nullptr;

static void cbFramebufferSize(GLFWwindow*, int w, int h){ if(gEngine) gEngine->onResize(w,h); }
static void cbMouseButton(GLFWwindow*, int b, int a, int m){ if(gEngine) gEngine->onMouseButton(b,a,m); }
static void cbCursorPos(GLFWwindow*, double x, double y){ if(gEngine) gEngine->onCursorPos(x,y); }
static void cbScroll(GLFWwindow*, double dx, double dy){ if(gEngine) gEngine->onScroll(dx,dy); }
static void cbKey(GLFWwindow*, int k, int s, int a, int m){ if(gEngine) gEngine->onKey(k,s,a,m); }

static inline std::string trim(const std::string& s){
    size_t a=s.find_first_not_of(" \t\r\n"); if(a==std::string::npos) return "";
    size_t b=s.find_last_not_of(" \t\r\n"); return s.substr(a,b-a+1);
}

void Engine::loadTurbinConfDefaults(const std::string& path){
    std::ifstream f(path);
    if(!f) return;
    std::string line;
    auto kv = [](const std::string& ln, std::string& k, std::string& v){
        auto p = ln.find('='); if(p==std::string::npos) return false;
        k = trim(ln.substr(0,p)); v = trim(ln.substr(p+1)); return true;
    };
    while(std::getline(f,line)){
        if(line.empty() || line[0]=='#' || line[0]==';') continue;
        std::string k,v; if(!kv(line,k,v)) continue;
        if(k=="rpm") turbine.params.rpm = std::atof(v.c_str());
        else if(k=="blades") turbine.params.blades = std::atoi(v.c_str());
        else if(k=="shaftLength") turbine.params.shaftLength = std::atof(v.c_str());
        else if(k=="shaftRadius") turbine.params.shaftRadius = std::atof(v.c_str());
        else if(k=="hubRadius") turbine.params.hubRadius = std::atof(v.c_str());
        else if(k=="bladeLength") turbine.params.bladeLength = std::atof(v.c_str());
        else if(k=="bladeRootWidth") turbine.params.bladeRootWidth = std::atof(v.c_str());
        else if(k=="bladeTipWidth") turbine.params.bladeTipWidth = std::atof(v.c_str());
        else if(k=="bladeThickness") turbine.params.bladeThickness = std::atof(v.c_str());
    }
}

bool Engine::init(const AppConfig& cfgIn){
    cfg = cfgIn;
    if(!glfwInit()){ std::fprintf(stderr,"GLFW init fail\n"); return false; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // Start maximized for better UX
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

    GLFWwindow* win = glfwCreateWindow(cfg.width, cfg.height, cfg.title.c_str(), nullptr, nullptr);
    if(!win){ std::fprintf(stderr,"GLFW window fail\n"); glfwTerminate(); return false; }
    window = win;
    glfwMakeContextCurrent(win);
    // Fallback: ensure maximized
    glfwMaximizeWindow(win);

    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK){ std::fprintf(stderr,"GLEW init fail\n"); return false; }
    glfwSwapInterval(cfg.vsync?1:0);

    gEngine = this;
    glfwSetFramebufferSizeCallback(win, cbFramebufferSize);
    glfwSetMouseButtonCallback(win, cbMouseButton);
    glfwSetCursorPosCallback(win, cbCursorPos);
    glfwSetScrollCallback(win, cbScroll);
    glfwSetKeyCallback(win, cbKey);

    glEnable(0x0B71/*GL_DEPTH_TEST*/);
    glEnable(0x0BE2/*GL_BLEND*/);
    glBlendFunc(0x0302/*SRC_ALPHA*/, 0x0303/*ONE_MINUS_SRC_ALPHA*/);

    if(!renderer.init(UITheme::Crimson())) return false;

    // Read turbin.conf as defaults
    loadTurbinConfDefaults("turbin.conf");
    cfg.rotate = true;
    turbine.params.rpm = 1250.0f;
    turbine.params.current_rpm = cfg.rotate ? turbine.params.rpm : 0.0f;
    turbine.rebuild();

    srand((unsigned int)time(NULL));
    gWindStrengthPhase = (rand() / (float)RAND_MAX) * 2.f * 3.14159f;
    gWindSpeedPhase = (rand() / (float)RAND_MAX) * 2.f * 3.14159f;
    gWindJitterPhase = (rand() / (float)RAND_MAX) * 2.f * 3.14159f;
    if (cfg.rotate) {
        gWindInterpolant = 1.0f;
    }

    timer.last = glfwGetTime();
    // Initial camera looking slightly from above at origin
    cam.target = {0,0,0};
    cam.distance = 4.0f;
    cam.yaw = 0.7f;
    cam.pitch = 0.35f;

#ifdef USE_IMGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL((GLFWwindow*)window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
    SetupDefaultFonts(16.0f);
    // Fonts configured by SetupDefaultFonts()

    NG_BuildDinosaur(graph);
#endif
    
    LoadEngineConf("engine.conf");
    {
      bool exists = (std::ifstream("engine.conf").good());
      if(!exists){
          // First run: write defaults
          SaveEngineConf("engine.conf");
      }
    }
    Logf("Engine %s initialized.", kAppVersion);

    // Player (for F8 top-down): build arrow mesh and defaults
    // Use a streamlined wedge prism for better top-down visibility
    playerMesh = Procedural::makeWedgePrism(1.2f, 0.6f, 0.25f);
    bushMesh = Procedural::makeBox(0.6f, 0.3f, 0.6f);
    cylinderMesh = Procedural::makeCylinder(0.35f, 0.8f, 28, true);
    players.clear(); bushes.clear(); aiControllers.clear();
    // Default manual player (gold) for user control
    players.push_back({});
    players.back().name = "Rabbit";
    // Place player at bottom-center within ortho frame (±6), slightly above grid, facing turbine
    players.back().pos = {0.0f, 0.25f, -5.5f};
    players.back().yaw = 0.0f;
    players.back().color = {0.95f,0.85f,0.2f};
    players.back().aiIndex = -1;
    selectedPlayer = 0;
    // No default bushes; start with a clean map
    // Removed saved networks scanning; Simulator Map no longer manages AI models
    return true;
}

void Engine::update(){
    double t = glfwGetTime(); timer.update(t);

    // Smooth wind transition
    const float kWIND_TRANSITION_SPEED = 1.5f; // seconds to go from 0 to 1
    if (cfg.rotate) {
        gWindInterpolant += (float)timer.dt / kWIND_TRANSITION_SPEED;
        if (gWindInterpolant > 1.0f) gWindInterpolant = 1.0f;
    } else {
        gWindInterpolant -= (float)timer.dt / kWIND_TRANSITION_SPEED;
        if (gWindInterpolant < 0.0f) gWindInterpolant = 0.0f;
    }

    // Update wind parameter phases for oscillation
    gWindStrengthPhase += 0.5f * (float)timer.dt;
    gWindSpeedPhase += 0.4f * (float)timer.dt;
    gWindJitterPhase += 0.6f * (float)timer.dt;

    if (gStrengthSineEnabled) {
        gWind.strength = 150.0f + 150.0f * sin(gWindStrengthPhase);
    }
    if (gSpeedSineEnabled) {
        gWind.speed = 2.5f + 2.5f * sin(gWindSpeedPhase);
    }
    if (gJitterSineEnabled) {
        gWind.jitter = 0.25f + 0.25f * sin(gWindJitterPhase);
    }

    if(cfg.rotate){
        float rps = turbine.params.rpm / 60.0f;
        float radps = rps * 6.28318530718f;
        rotorAngle += (float)(radps * timer.dt);
    }
    // Update manual/AI movement always so arrows work in 3D too
    updateTopDownPlayers((float)timer.dt);
    if(topDown2D){
        if(cameraFollow){ int sp = (selectedPlayer>=0 && selectedPlayer<(int)players.size())? selectedPlayer : 0; if(!players.empty()) topDownCenterTarget = players[sp].pos; }
        float k = std::min(1.0f, (float)timer.dt * 6.0f);
        topDownCenter.x += (topDownCenterTarget.x - topDownCenter.x) * k;
        topDownCenter.z += (topDownCenterTarget.z - topDownCenter.z) * k;
        pushPlayerCoordsToInputs();
        applyInputMappings();
    }
    processCamera();
}

Mat4 Engine::proj() const {
    int w,h; glfwGetFramebufferSize((GLFWwindow*)window, &w, &h);
    float aspect = (h>0) ? (float)w/(float)h : 1.0f;
    if(topDown2D){
        float base = 6.0f; // world units half-extent vertically
        float left,right,bottom,top;
        if(aspect >= 1.0f){
            top = base; bottom = -base; right = base*aspect; left = -right;
        } else {
            right = base; left = -base; top = base/aspect; bottom = -top;
        }
        return Mat4::ortho(left,right,bottom,top,-50.0f,50.0f);
    }
    return Mat4::perspective(deg2rad(55.0f), aspect, 0.05f, 100.0f);
}
Mat4 Engine::view() const {
    if(topDown2D){
        // Look straight down from +Y towards current center; use Z-up so +Z maps to top of screen
        Vec3 c = topDownCenter;
        return Mat4::lookAt({c.x, 10.0f, c.z + 0.001f}, {c.x, 0.0f, c.z}, {0.0f, 0.0f, +1.0f});
    }
    return cam.view();
}

#ifdef USE_IMGUI
// Keep track of HUD rectangle for overlay hole
static ImVec2 gHUDMin(0,0), gHUDMax(0,0);
static bool gHUDHasRect = false;
static bool gShowExitConfirm = false;

// Overlay training HUD in the center while training
static void DrawTrainingHUD(){
    if(!gTrain.active) return;
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 sz = io.DisplaySize;
    ImVec2 winSz(560, 360);
    ImGui::SetNextWindowPos(ImVec2((sz.x - winSz.x)*0.5f, (sz.y - winSz.y)*0.5f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(winSz, ImGuiCond_Always);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
    ImGui::Begin("Training HUD", nullptr, flags);
    ImGui::TextColored(ImVec4(0.9f,0.95f,1.0f,1), "Training in progress");
    ImGui::Separator();
    float last = gTrain.mseHistory.empty()? 0.0f : gTrain.mseHistory.back();
    ImGui::Text("Epoch: %d / %d", gTrain.epoch, gTrain.totalEpochs);
    ImGui::Text("MSE:   %.6f  (target: %s)", last, gTrain.targetMSE>0?std::to_string(gTrain.targetMSE).c_str():"-");
    ImGui::Text("Avg epoch time: %.2f ms", gTrain.avgEpochMs);
    double elapsed = (gTrain.startTime>0.0) ? (nowSeconds()-gTrain.startTime) : 0.0;
    ImGui::Text("Elapsed: %.1f s", elapsed);
    ImGui::Spacing();
    ImGui::PlotLines("MSE", gTrain.mseHistory.data(), (int)gTrain.mseHistory.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(-1,200));
            // Axes and ticks for MSE plot
            {
                ImDrawList* dl = ImGui::GetWindowDrawList();
                ImVec2 p0 = ImGui::GetItemRectMin();
                ImVec2 p1 = ImGui::GetItemRectMax();
                // y axis at left, x axis at bottom
                dl->AddLine(ImVec2(p0.x, p0.y), ImVec2(p0.x, p1.y), ImGui::GetColorU32(ImVec4(0.8f,0.8f,0.8f,0.6f)), 1.0f);
                dl->AddLine(ImVec2(p0.x, p1.y), ImVec2(p1.x, p1.y), ImGui::GetColorU32(ImVec4(0.8f,0.8f,0.8f,0.6f)), 1.0f);
                // ticks
                const int xticks = 10;
                const int yticks = 5;
                for(int i=0;i<=xticks;i++){
                    float t = (float)i/(float)xticks;
                    float x = p0.x + t*(p1.x-p0.x);
                    dl->AddLine(ImVec2(x, p1.y), ImVec2(x, p1.y-5.0f), ImGui::GetColorU32(ImVec4(0.7f,0.7f,0.7f,0.6f)), 1.0f);
                }
                // Estimate y-range from recent history
                float ymin = 1e9f, ymax = -1e9f;
                int n = (int)gTrain.mseHistory.size();
                int wnd = n;
                for(int i=0;i<wnd;i++){ float v = gTrain.mseHistory[i]; if(v<ymin) ymin=v; if(v>ymax) ymax=v; }
                if(!(ymax>ymin)) { ymin = 0.0f; ymax = 1.0f; }
                for(int i=0;i<=yticks;i++){
                    float t = (float)i/(float)yticks;
                    float y = p1.y - t*(p1.y-p0.y);
                    dl->AddLine(ImVec2(p0.x, y), ImVec2(p0.x+5.0f, y), ImGui::GetColorU32(ImVec4(0.7f,0.7f,0.7f,0.6f)), 1.0f);
                    // optional labels: small text with value
                    float v = ymin + t*(ymax-ymin);
                    char buf[32]; std::snprintf(buf, sizeof(buf), "%.3f", v);
                    dl->AddText(ImVec2(p0.x+6.0f, y-8.0f), ImGui::GetColorU32(ImVec4(0.7f,0.7f,0.7f,0.9f)), buf);
                }
            }
    // Progress bar under the plot: percent of epochs completed
    {
        float pct = 0.0f;
        if(gTrain.totalEpochs > 0) pct = std::clamp(gTrain.epoch / (float)std::max(1, gTrain.totalEpochs), 0.0f, 1.0f);
        ImGui::Spacing();
        char pbuf[32]; std::snprintf(pbuf, sizeof(pbuf), "%d%%", (int)std::round(pct*100.0f));
        ImGui::ProgressBar(pct, ImVec2(-1, 0), pbuf);
    }
        
    ImGui::End();
}
#endif

void Engine::render(){
    int w,h; glfwGetFramebufferSize((GLFWwindow*)window, &w, &h);
    glViewport(0,0,w,h);
    glClearColor(renderer.theme.bg.x, renderer.theme.bg.y, renderer.theme.bg.z, 1.0f);
    glClear(0x00004000/*GL_COLOR_BUFFER_BIT*/ | 0x00000100/*GL_DEPTH_BUFFER_BIT*/);

    Mat4 P = proj();
    Mat4 V = view();
    Mat4 VP = P * V;

    renderer.drawGrid(VP);

    Mat4 Mshaft = Mat4::translate({0,0,0});
    renderer.drawMesh(turbine.shaft, VP*Mshaft, Mshaft, renderer.theme.mesh, 0.05f);

    // Draw full rotor as one rotating assembly around Y axis at origin
    Mat4 Mrotor = Mat4::rotateY(rotorAngle);
    renderer.drawMesh(turbine.rotor, VP*Mrotor, Mrotor, renderer.theme.accent, 0.4f);
    // Draw static pipes
    Mat4 Mpipes = Mat4::translate({0,0,0});
    renderer.drawMesh(turbine.pipes, VP*Mpipes, Mpipes, renderer.theme.mesh, 0.2f);
    // Steam lines (stylized) - always visible
    renderer.drawLines(turbine.schematicLines, VP, renderer.theme.accent, 2.0f);
    // Draw meshing outer gears (counter-rotating), translated on X (left and right)
    float ratioL = (turbine.outerTeeth>0)? (turbine.rotorTeeth / (float)turbine.outerTeeth) : 1.0f;
    Mat4 MouterL = Mat4::translate({turbine.outerGearOffset, 0, 0}) * Mat4::rotateY(-rotorAngle*ratioL);
    renderer.drawMesh(turbine.outerGear, VP*MouterL, MouterL, renderer.theme.mesh, 0.6f);
    float ratioR = (turbine.outerTeethR>0)? (turbine.rotorTeeth / (float)turbine.outerTeethR) : 1.0f;
    Mat4 MouterR = Mat4::translate({turbine.outerGearOffsetR, 0, 0}) * Mat4::rotateY(rotorAngle*ratioR);
    renderer.drawMesh(turbine.outerGearR, VP*MouterR, MouterR, renderer.theme.mesh, 0.6f);

    if(cfg.showSchematic){
        renderer.drawLines(turbine.schematicLines, VP, renderer.theme.grid, 1.0f);
    }

    // Draw players, bushes, and placed shapes
    if(topDown2D){
        for(const auto& b : bushes){
            Mat4 M = Mat4::translate(b.pos);
            renderer.drawMesh(bushMesh, VP*M, M, {0.2f,0.7f,0.25f}, 0.1f);
        }
        for(const auto& p : players){
            Mat4 M = Mat4::translate(p.pos) * Mat4::rotateY(p.yaw) * Mat4::scale({1.4f,1.4f,1.4f});
            renderer.drawMesh(playerMesh, VP*M, M, p.color, 0.2f);
        }
        for(size_t i=0;i<shapes.size();++i){
            const auto& sEnt = shapes[i];
            Mat4 M = Mat4::translate(sEnt.pos) * Mat4::rotateY(sEnt.yaw) * Mat4::scale(sEnt.scale);
            const Mesh* mesh = &playerMesh;
            if(sEnt.type == ShapeType::Box) mesh = &bushMesh;
            else if(sEnt.type == ShapeType::Cylinder) mesh = &cylinderMesh;
            // Flat color to match chosen picker exactly
            renderer.drawMeshFlat(*mesh, VP*M, sEnt.color);
            if((int)i == selectedShape){
                Mat4 Mh = Mat4::translate(sEnt.pos) * Mat4::rotateY(sEnt.yaw) * Mat4::scale({sEnt.scale.x*1.12f,sEnt.scale.y*1.12f,sEnt.scale.z*1.12f});
                renderer.drawMeshFlat(*mesh, VP*Mh, renderer.theme.accent);
            }
            // Overlay shape name at center (F8 only)
            ImDrawList* fg = ImGui::GetForegroundDrawList();
            int w,h; glfwGetFramebufferSize((GLFWwindow*)window, &w, &h);
            auto projectToScreen = [&](const Vec3& wp){
                float x=wp.x, y=wp.y, z=wp.z;
                float cx = VP.m[0]*x + VP.m[4]*y + VP.m[8]*z + VP.m[12];
                float cy = VP.m[1]*x + VP.m[5]*y + VP.m[9]*z + VP.m[13];
                float cw = VP.m[3]*x + VP.m[7]*y + VP.m[11]*z + VP.m[15];
                if(cw==0) cw=1e-6f; float ndcX = cx/cw, ndcY = cy/cw;
                float sx = (ndcX*0.5f + 0.5f) * (float)w;
                float sy = (1.0f - (ndcY*0.5f + 0.5f)) * (float)h;
                return ImVec2(sx, sy);
            };
            ImVec2 sp2 = projectToScreen(sEnt.pos);
            char nb[128]; std::snprintf(nb, sizeof(nb), "%s", sEnt.name.c_str());
            ImVec2 ts = ImGui::CalcTextSize(nb);
            ImVec2 p0(sp2.x - ts.x*0.5f - 4.0f, sp2.y - ts.y*0.5f - 2.0f);
            ImVec2 p1(sp2.x + ts.x*0.5f + 4.0f, sp2.y + ts.y*0.5f + 2.0f);
            fg->AddRectFilled(p0, p1, IM_COL32(0,0,0,200), 3.0f);
            fg->AddText(ImVec2(p0.x+4.0f, p0.y+2.0f), IM_COL32(255,255,255,255), nb);
        }
    }
    else {
        // 3D view: also draw players so they are visible
        for(const auto& p : players){
            Mat4 M = Mat4::translate(p.pos) * Mat4::rotateY(p.yaw) * Mat4::scale({1.4f,1.4f,1.4f});
            renderer.drawMesh(playerMesh, VP*M, M, p.color, 0.2f);
        }
        for(size_t i=0;i<shapes.size();++i){
            const auto& sEnt = shapes[i];
            Mat4 M = Mat4::translate(sEnt.pos) * Mat4::rotateY(sEnt.yaw) * Mat4::scale(sEnt.scale);
            const Mesh* mesh = &playerMesh;
            if(sEnt.type == ShapeType::Box) mesh = &bushMesh;
            else if(sEnt.type == ShapeType::Cylinder) mesh = &cylinderMesh;
            renderer.drawMeshFlat(*mesh, VP*M, sEnt.color);
            if((int)i == selectedShape){
                Mat4 Mh = Mat4::translate(sEnt.pos) * Mat4::rotateY(sEnt.yaw) * Mat4::scale({sEnt.scale.x*1.12f,sEnt.scale.y*1.12f,sEnt.scale.z*1.12f});
                renderer.drawMeshFlat(*mesh, VP*Mh, renderer.theme.accent);
            }
        }
    }

#ifdef USE_IMGUI
    // Update OS window title with theme
    {
        char wtitle[256]; std::snprintf(wtitle, sizeof(wtitle), "neuralT %s - Theme: %s", kAppVersion, renderer.theme.name.c_str());
        glfwSetWindowTitle((GLFWwindow*)window, wtitle);
    }
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();

    // Auto-layout windows each frame (deterministic positions)
    {
        ImGuiIO& io = ImGui::GetIO();
        ImVec2 disp = io.DisplaySize;

        // Left top: Neural Editor (big)
        float neW = std::max(400.0f, disp.x * 0.62f);
        float neH = std::max(300.0f, disp.y * 0.62f);
        if(gShowEditorWindow){
            bool openNe = true;
            graph.drawCanvas(openNe, neW - 20.0f, neH - 20.0f);
        }

        // Top-right: NeuralT control panel
        ImGuiCond layCond = gForceRelayoutWindows ? ImGuiCond_Always : ImGuiCond_FirstUseEver;
        ImGui::SetNextWindowPos(ImVec2(disp.x - 360.0f, 10.0f), layCond);
        ImGui::SetNextWindowSize(ImVec2(350.0f, 270.0f), layCond);
        if(cfg.showUI && gShowNeuralXPanel){
            ImGui::Begin("Turbine [F2]", &gShowNeuralXPanel, ImGuiWindowFlags_NoCollapse);
            char title[128];
            std::snprintf(title, sizeof(title), "NeuralT %s | dt: %.3f ms", kAppVersion, timer.dt*1000.0);
            ImGui::TextUnformatted(title);
            ImGui::Text("Total Neurons: %zu", graph.nodes.size());
            ImGui::Checkbox("VSync (V)", &cfg.vsync);
            ImGui::SameLine(); ImGui::Checkbox("Rotate (R)", &cfg.rotate);
            ImGui::Checkbox("Schematic (G)", &cfg.showSchematic);
            ImGui::SliderFloat("RPM", &turbine.params.rpm, 0.0f, 4000.0f);
            int blades = turbine.params.blades;
            if(ImGui::SliderInt("Blades", &blades, 1, 10)){ turbine.params.blades = blades; turbine.rebuild(); }
            if(ImGui::Button("Reset Turbine Settings")){
                turbine.params.rpm = 1250.0f;
                gWind.strength = 150.0f;
                gWind.speed = 1.0f;
                gStrengthSineEnabled = false;
                gSpeedSineEnabled = false;
                gJitterSineEnabled = true;
                turbine.params.current_rpm = cfg.rotate ? turbine.params.rpm : 0.0f;
                Logf("Turbine settings reset: RPM=1250, Strength=150, Speed=1.0, Jitter mod=on");
            }
            ImGui::Separator();
            ImGui::Text("Wind Effect");

            auto draw_sine_plot = [](const char* id, float phase) {
                ImGui::PushID(id);
                ImVec2 plotSize(ImGui::GetContentRegionAvail().x - 60.f, 30);
                ImGui::InvisibleButton("plot", plotSize);
                ImDrawList* drawList = ImGui::GetWindowDrawList();
                ImVec2 p0 = ImGui::GetItemRectMin();
                ImVec2 p1 = ImGui::GetItemRectMax();

                drawList->AddRectFilled(p0, p1, IM_COL32(30, 30, 35, 255));
                drawList->AddRect(p0, p1, IM_COL32(150, 150, 160, 255));

                const int num_segments = 50;
                ImVec2 points[num_segments + 1];
                for (int i = 0; i <= num_segments; ++i) {
                    float t = (float)i / (float)num_segments;
                    float x = p0.x + t * (p1.x - p0.x);
                    float y = p0.y + (p1.y - p0.y) / 2.0f - (p1.y - p0.y) / 2.2f * sin(t * 6.0f + phase);
                    points[i] = ImVec2(x, y);
                }
                for (int i = 0; i < num_segments; ++i) {
                    drawList->AddLine(points[i], points[i + 1], IM_COL32(255, 80, 80, 255), 1.5f);
                }
                ImGui::PopID();
            };

            if (gStrengthSineEnabled) ImGui::BeginDisabled();
            ImGui::SliderFloat("Strength", &gWind.strength, 0.0f, 300.0f, "%.2f");
            if (gStrengthSineEnabled) ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Checkbox("##StrengthSine", &gStrengthSineEnabled);
            draw_sine_plot("StrengthPlot", gWindStrengthPhase);

            if (gSpeedSineEnabled) ImGui::BeginDisabled();
            ImGui::SliderFloat("Speed", &gWind.speed, 0.0f, 5.0f, "%.2f");
            if (gSpeedSineEnabled) ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Checkbox("##SpeedSine", &gSpeedSineEnabled);
            draw_sine_plot("SpeedPlot", gWindSpeedPhase);

            if (gJitterSineEnabled) ImGui::BeginDisabled();
            ImGui::SliderFloat("Jitter", &gWind.jitter, 0.0f, 0.5f, "%.2f");
            if (gJitterSineEnabled) ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Checkbox("##JitterSine", &gJitterSineEnabled);
            draw_sine_plot("JitterPlot", gWindJitterPhase);

            ImGui::Separator();
            ImGui::Text("Live Feed Wave");
            if(ImGui::Button(gLiveFeedWaveActive ? "Disable Wave (W)" : "Enable Wave (W)")){
                gLiveFeedWaveActive = !gLiveFeedWaveActive;
                if(gLiveFeedWaveActive){
                    gLiveFeedCurrentInput = 0;
                    gLiveFeedInputTimer = 0.0f;
                    gLiveFeedWavePhase = 0.0f;
                    Logf("Live feed wave effect activated");
                } else {
                    Logf("Live feed wave effect deactivated");
                }
            }
            if(gLiveFeedWaveActive){
                ImGui::SliderFloat("Wave Speed", &gLiveFeedWaveSpeed, 0.5f, 5.0f, "%.1f");
                ImGui::SliderFloat("Input Delay", &gLiveFeedInputDelay, 0.1f, 1.0f, "%.2f s");
                ImGui::Text("Current Input: %d", gLiveFeedCurrentInput + 1);
            }
            
            ImGui::Separator();
            ImGui::Text("Input Controls");
            
            // Random Live Feed toggle button
            static bool randomLiveFeedActive = false;
            if(ImGui::Button(randomLiveFeedActive ? "Disable Random Feed (Q)" : "Enable Random Feed (Q)")){
                randomLiveFeedActive = !randomLiveFeedActive;
                int inputCount = 0;
                for(const auto& L : graph.layers){
                    if(L.kind != LayerKind::Input) continue;
                    for(int idx : L.nodeIdx){
                        if(idx < 0 || idx >= (int)graph.nodes.size()) continue;
                        auto& n = graph.nodes[idx];
                        if(!n.isInput) continue;
                        
                        if(randomLiveFeedActive){
                            // Set random amplitude (0.1 to 1.0)
                            n.amp = 0.1f + (rand() / (float)RAND_MAX) * 0.9f;
                            // Set random speed (0.1 to 1.0 Hz)
                            n.speed = 0.1f + (rand() / (float)RAND_MAX) * 0.9f;
                            // Enable live feed
                            n.live = true;
                            // Reset phase for immediate effect
                            n.phase = 0.0f;
                        } else {
                            // Disable live feed
                            n.live = false;
                            n.speed = 0.0f;
                        }
                        inputCount++;
                    }
                }
                if(randomLiveFeedActive){
                    Logf("Random live feed applied to %d input neurons", inputCount);
                } else {
                    Logf("Random live feed disabled for %d input neurons", inputCount);
                }
            }
            
            // Zero All Inputs button
            if(ImGui::Button("Zero All Inputs")){
                int inputCount = 0;
                for(const auto& L : graph.layers){
                    if(L.kind != LayerKind::Input) continue;
                    for(int idx : L.nodeIdx){
                        if(idx < 0 || idx >= (int)graph.nodes.size()) continue;
                        auto& n = graph.nodes[idx];
                        if(!n.isInput) continue;
                        
                        // Set all input values to zero
                        n.value = 0.0f;
                        // Disable live feed
                        n.live = false;
                        n.speed = 0.0f;
                        n.phase = 0.0f;
                        inputCount++;
                    }
                }
                // Also disable global live inputs
                graph.liveInputs = false;
                // Disable wave effect
                gLiveFeedWaveActive = false;
                randomLiveFeedActive = false;
                Logf("All %d input neurons set to zero and live feeds disabled", inputCount);
            }

            ImGui::End();
        }

        // Detached Live Feed window (F6)
        if(gShowLiveFeedWindow){
            ImGui::SetNextWindowSize(ImVec2(700, 320), ImGuiCond_FirstUseEver);
            if(ImGui::Begin("Live Feed - Inputs [F6]", &gShowLiveFeedWindow, ImGuiWindowFlags_NoCollapse)){
                if(!graph.layers.empty()){
                    const Layer& L = graph.layers[std::clamp(graph.currentLayer, 0, (int)graph.layers.size()-1)];
                    ImGui::Text("Live Feed - %s (neurons: %d, activation: %s)", L.name.c_str(), (int)L.nodeIdx.size(), ActName(L.act));
                    if(ImGui::BeginTable("lf_win", 6, ImGuiTableFlags_Resizable|ImGuiTableFlags_RowBg|ImGuiTableFlags_BordersInnerV)){
                        ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 40.f);
                        ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 80.f);
                        ImGui::TableSetupColumn("Speed(Hz)", ImGuiTableColumnFlags_WidthFixed, 90.f);
                        ImGui::TableSetupColumn("In", ImGuiTableColumnFlags_WidthFixed, 40.f);
                        ImGui::TableSetupColumn("Out", ImGuiTableColumnFlags_WidthFixed, 45.f);
                        ImGui::TableHeadersRow();
                        for(int idx : L.nodeIdx){
                            Neuron& n = graph.nodes[idx];
                            int inCnt=0, outCnt=0;
                            for(auto& e: graph.edges){ if(e.b==n.id) ++inCnt; if(e.a==n.id) ++outCnt; }
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0); ImGui::Text("%d", n.id);
                            ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(n.label.c_str());
                            ImGui::TableSetColumnIndex(2);
                            float v = n.value; ImVec4 c = ImVec4(1.0f - v, 0.2f + 0.8f*v, 0.15f, 1.0f);
                            ImGui::TextColored(c, "%.3f", v);
                            ImGui::TableSetColumnIndex(3);
                            if(n.isInput) ImGui::Text("%.2f", n.speed); else ImGui::TextDisabled("-");
                            ImGui::TableSetColumnIndex(4); ImGui::Text("%d", inCnt);
                            ImGui::TableSetColumnIndex(5); ImGui::Text("%d", outCnt);
                        }
                        ImGui::EndTable();
                    }
                } else {
                    ImGui::TextDisabled("No layers to display.");
                }
            }
            ImGui::End();
        }

        // Left-bottom: Log window
        float logW = neW - 20.0f;
        float logH = disp.y - neH - 30.0f;
        if(logH < 150.0f) logH = 150.0f;
        if(gShowLogWindow){
        ImGui::SetNextWindowPos(ImVec2(10.0f, neH), layCond);
        ImGui::SetNextWindowSize(ImVec2(logW, logH), layCond);
            ImGui::Begin("Log [F3]", &gShowLogWindow, ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoSavedSettings);
            ImGui::Text("Lessons: %d", gLessonsCount);
            ImGui::Separator();
            ImGui::BeginChild("log_scroller", ImVec2(0, -28), true, ImGuiWindowFlags_HorizontalScrollbar);
            for(const auto& s : gLogLines) ImGui::TextUnformatted(s.c_str());
            if(gLogAutoScroll) ImGui::SetScrollHereY(1.0f);
            ImGui::EndChild();
            ImGui::Checkbox("Auto scroll", &gLogAutoScroll);
            ImGui::SameLine();
            if(ImGui::Button("Clear")) gLogLines.clear();
            ImGui::End();
        }

        // Right-bottom: Training parameters
        float paramsW = disp.x - (10.0f + logW) - 20.0f;
        if(paramsW < 320.0f) paramsW = 320.0f;
        float paramsH = std::max(220.0f, disp.y * 0.38f);
        if(gShowParamsWindow){
        ImGui::SetNextWindowPos(ImVec2(disp.x - paramsW - 10.0f, disp.y - paramsH - 10.0f), layCond);
        ImGui::SetNextWindowSize(ImVec2(paramsW, paramsH), layCond);
            ImGui::Begin("Training parameters [F4]", &gShowParamsWindow, ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoSavedSettings);
        if(!gTrain.active){
            ImGui::SliderInt("Epochs", &gTrain.totalEpochs, 10, 5000);
            ImGui::SliderFloat("Learning rate", &gTrain.lr, 1e-5f, 1.0f, "%.6f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("L1", &gTrain.l1, 0.0f, 0.1f, "%.6f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("L2", &gTrain.l2, 0.0f, 0.1f, "%.6f", ImGuiSliderFlags_Logarithmic);
            ImGui::InputFloat("Target MSE (<=0 off)", &gTrain.targetMSE, 0.0f, 0.0f, "%.6f");
            ImGui::Separator();
            // Row 1: Examples (Rabbit/Predator), Help
            if(ImGui::Button("Rabbit Example")){
                // Build 6-10-8-5 network for rabbit scenario
                NG_BuildTopology(graph, 6, {10,8,5}, 5);
                // Label inputs/outputs with descriptions
                {
                    static const char* kInputs[6]  = {
                        "CarrotProximity", "FoxProximity", "HungerLevel",
                        "ThreatLevel", "BushProximity", "SafetyLevel"
                    };
                    static const char* kInputDescs[6] = {
                        "Proximity to nearest carrot (1=close, 0=far)",
                        "Proximity to nearest fox (1=close, 0=far)",
                        "Current hunger level (1=very hungry, 0=full)",
                        "Perceived threat level (1=high danger, 0=safe)",
                        "Proximity to nearest bush (1=close, 0=far)",
                        "Current safety level (1=very safe, 0=exposed)"
                    };
                    static const char* kActions[5] = {
                        "Flee", "Hide", "MoveToFood", "Eat", "Rest"
                    };
                    static const char* kActionDescs[5] = {
                        "Action to flee from danger",
                        "Action to hide in nearby cover",
                        "Action to move towards food",
                        "Action to eat available food",
                        "Action to rest and recover"
                    };
                    if(!graph.layers.empty() && graph.layers.front().kind==LayerKind::Input){
                        auto& L = graph.layers.front();
                        for(size_t i=0;i<L.nodeIdx.size() && i<6;i++){ 
                            graph.nodes[L.nodeIdx[i]].label = std::string(kInputs[i]);
                            graph.nodes[L.nodeIdx[i]].desc = std::string(kInputDescs[i]);
                        }
                    }
                    if(!graph.layers.empty()){
                        auto& L = graph.layers.back();
                        if(L.kind==LayerKind::Output){
                            for(size_t i=0;i<L.nodeIdx.size() && i<5;i++){ 
                                graph.nodes[L.nodeIdx[i]].label = std::string(kActions[i]);
                                graph.nodes[L.nodeIdx[i]].desc = std::string(kActionDescs[i]);
                            }
                        }
                    }
                }
                // Randomize hidden layers for a fresh start
                NG_RandomizeHidden(graph, 0.8f, 0.3f);
                // Training params
                gTrain.lr = 0.01f;
                gTrain.l1 = 0.0f;
                gTrain.l2 = 0.0001f;
                gTrain.totalEpochs = 3000;
                gTrain.targetMSE = 0.02f;
                // Dataset from specification (6 inputs, 5 outputs one-hot) with descriptions
                gDataset.clear();
                auto make_sample = [&](const std::array<float,6>& in, const std::array<float,5>& out, const std::string& desc){
                    TrainSample s; s.in.assign(in.begin(), in.end()); s.out.assign(out.begin(), out.end()); s.desc = desc; return s;
                };
                gDataset.push_back(make_sample({0.0f, 0.9f, 0.0f, 0.8f, 0.3f, 0.6f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, "Fox nearby, no food â†’ Flee"));
                gDataset.push_back(make_sample({0.0f, 0.8f, 0.0f, 0.7f, 0.9f, 0.5f}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f}, "Fox nearby, bush close â†’ Hide"));
                gDataset.push_back(make_sample({0.0f, 0.0f, 0.9f, 0.0f, 0.2f, 0.1f}, {0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, "Hungry, sees carrot, no threat â†’ Move to food"));
                gDataset.push_back(make_sample({0.8f, 0.0f, 0.9f, 0.0f, 0.1f, 0.2f}, {0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, "At carrot, hungry, safe â†’ Eat"));
                gDataset.push_back(make_sample({0.1f, 0.0f, 0.0f, 0.0f, 0.2f, 0.9f}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, "Full, no threat â†’ Rest"));
                gDataset.push_back(make_sample({0.7f, 0.8f, 0.6f, 0.7f, 0.2f, 0.3f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, "Fox and food nearby, low hunger â†’ Flee"));
                gDataset.push_back(make_sample({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.8f}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, "Tired, full, no signals â†’ Rest"));
                gDataset.push_back(make_sample({0.0f, 0.2f, 0.9f, 0.1f, 0.1f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, "Very hungry, sees carrot, fox far â†’ Move to food"));
                gDataset.push_back(make_sample({0.0f, 0.9f, 0.0f, 0.7f, 0.0f, 0.2f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, "Low energy, smells fox, no shelter â†’ Flee"));
                gDataset.push_back(make_sample({0.6f, 0.0f, 0.9f, 0.0f, 0.1f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, "Full, sees carrot, no fox â†’ Rest"));
                gDataset.push_back(make_sample({0.0f, 0.9f, 0.0f, 0.6f, 0.5f, 0.4f}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f}, "Strong fox smell, no shelter â†’ Hide"));
                gDataset.push_back(make_sample({0.0f, 0.2f, 0.0f, 0.4f, 0.9f, 0.6f}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f}, "Fox far, bush close â†’ Hide"));
                gDataset.push_back(make_sample({0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.6f}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, "Slightly hungry, safe â†’ Rest"));
                gDataset.push_back(make_sample({0.7f, 0.1f, 0.0f, 0.2f, 0.0f, 0.3f}, {0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, "Fox very far, smells carrot â†’ Move to food"));
                gDataset.push_back(make_sample({0.5f, 0.2f, 0.4f, 0.1f, 0.0f, 0.9f}, {0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, "Full, slight fox smell â†’ Eat (safe, enough energy)"));
                gDataset.push_back(make_sample({0.6f, 0.0f, 0.7f, 0.0f, 0.1f, 0.7f}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, "High energy, senses carrot and safety â†’ Rest"));
                gDataset.push_back(make_sample({0.0f, 0.0f, 0.7f, 0.0f, 0.8f, 0.3f}, {0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, "Carrot visible behind bush, no fox â†’ Move to food"));
                gLessonsCount = (int)gDataset.size();
                SaveEngineConf("engine.conf");
                Logf("Rabbit Example initialized: topology 6-10-8-5, lessons=%d", gLessonsCount);
            }
            ImGui::SameLine();
            if(ImGui::Button("Predator Example")){
                SetupPredatorExampleGraph();
            }
            ImGui::SameLine();
            if(ImGui::Button("Help (H)")) gShowHelp = !gShowHelp;
            // Row 2: Start, Train To Target, Save Config
            if(ImGui::Button("Start")){
                if(!gDataset.empty()){
                    // Break symmetry if weights look uniform (e.g., freshly rebuilt)
                    bool uniform = true; float ref = (!graph.edges.empty()? graph.edges.front().w : 0.0f);
                    for(const auto& e_ : graph.edges){ if(std::fabs(e_.w - ref) > 1e-6f){ uniform=false; break; } }
                    if(uniform) NG_RandomizeHidden(graph, 1.0f, 0.5f);
                    gTrain.active=true; gTrain.done=false; gTrain.epoch=0; gTrain.mseHistory.clear(); gTrain.startTime = nowSeconds(); gTrain.avgEpochMs=0.0;
                    Logf("Training started: epochs=%d, lr=%.6f, L1=%.6f, L2=%.6f, targetMSE=%.6f",
                         gTrain.totalEpochs, gTrain.lr, gTrain.l1, gTrain.l2, gTrain.targetMSE);
                }else{
                    Logf("Cannot start training: dataset is empty.");
                }
            }
            ImGui::SameLine();
            if(ImGui::Button("Train To Target (T)")){
                if(gDataset.empty()){
                    Logf("Cannot train to target: dataset is empty. idiot.");
                } else if(gTrain.targetMSE <= 0.0f){
                    Logf("Cannot train to target: set Target MSE > 0 first. dushebag.");
                } else {
                    bool uniform = true; float ref = (!graph.edges.empty()? graph.edges.front().w : 0.0f);
                    for(const auto& e_ : graph.edges){ if(std::fabs(e_.w - ref) > 1e-6f){ uniform=false; break; } }
                    if(uniform) NG_RandomizeHidden(graph, 1.0f, 0.5f);
                    gTrain.totalEpochs = std::max(gTrain.totalEpochs, 200000); // give plenty of room
                    gTrain.active=true; gTrain.done=false; gTrain.epoch=0; gTrain.mseHistory.clear(); gTrain.startTime = nowSeconds(); gTrain.avgEpochMs=0.0;
                    Logf("Train-to-target started: targetMSE=%.6f, epochs=%d, lr=%.6f", gTrain.targetMSE, gTrain.totalEpochs, gTrain.lr);
                }
            }
            ImGui::SameLine();
            if(ImGui::Button("Save Config")){ SaveEngineConf("engine.conf"); Logf("Config saved: engine.conf"); }
        }else{
            ImGui::TextDisabled("Training in progress...");
            if(ImGui::Button("Stop")){
                gTrain.active = false;
                gTrain.done = true;
                // Ensure any dimming overlay and HUD are cleared when stopping
                gShowTrainingWindow = false;
                gHUDHasRect = false;
                // Show result window until confirmed
                extern bool gCelebrationVisible; // already static above in same TU
                extern bool gTargetReachedVisual;
                extern bool gTrainInterrupted;
                gCelebrationVisible = true;
                gTargetReachedVisual = false;
                gTrainInterrupted = true;
                SaveEngineConf("engine.conf");
                Logf("Training stopped at epoch %d (MSE=%.6f).", gTrain.epoch, gTrain.lastMSE);
            }
        }
        ImGui::Separator();
        ImGui::Text("Dataset: %d lessons", gLessonsCount);
        ImGui::Text("IQ (total): %.0f", gIQTotal);
        if(gTrain.active) ImGui::Text("Session IQ gain: %.0f", gIQSessionGain);
        // Row 3: Add Lesson, Randomize Hidden, Delete All Lessons
        if(ImGui::Button("Add Lesson (M)")){
            std::vector<float> in,out; CollectIO(graph,in,out);
            gDataset.push_back({in,out});
            gLessonsCount=(int)gDataset.size();
            SaveEngineConf("engine.conf");
            Logf("Lesson added. Total=%d", gLessonsCount);
            // New lesson recorded: ensure outputs are not pinned by manual overrides
            gManualOutputs.clear();
            gUseManualOverrides = false;
        }
        ImGui::SameLine();
        if(ImGui::Button("Randomize Hidden (X)")){
            NG_RandomizeHidden(graph, 1.0f, 0.5f);
            Logf("Hidden layers randomized.");
        }
        ImGui::SameLine();
        if(ImGui::Button("Delete All Lessons")){
            gDataset.clear(); gLessonsCount = 0;
            SaveEngineConf("engine.conf");
            Logf("All lessons deleted. forever. it was shit anyway..");
        }
        // Row 4: Save/Load network
        if(ImGui::Button("New Network")){
            // New empty layers: Input + 3 Hidden + Output (no neurons)
            gDataset.clear();
            gLessonsCount = 0;
            gManualOutputs.clear();
            gUseManualOverrides = false;
            NG_BuildTopology(graph, /*inN*/0, std::vector<int>{0,0,0}, /*outN*/0);
            graph.currentLayer = 0;
            if(gEngine) gEngine->networkDirty = true;
            Logf("New empty network created (layers only).");
        }
        ImGui::SameLine();
        if(ImGui::Button("Save Network (S)")){
            std::error_code _ec; std::filesystem::create_directories("saved", _ec);
            // timestamped default name
            {
                std::time_t t = std::time(nullptr);
                std::tm tm{};
#if defined(_WIN32)
                localtime_s(&tm, &t);
#else
                localtime_r(&t, &tm);
#endif
                char buf[128];
                std::strftime(buf, sizeof(buf), "saved/network_%Y%m%d_%H%M%S.json", &tm);
                std::snprintf(gSaveName, sizeof(gSaveName), "%s", buf);
            }
            gShowSaveMenu = true;
        }
        ImGui::SameLine();
        if(ImGui::Button("Load Network (O)")){
            std::error_code _ec; std::filesystem::create_directories("saved", _ec);
            gShowLoadMenu = true;
        }
        // Lessons table with editable descriptions and actions
        ImGui::Separator();
        static std::vector<std::array<char,256>> lessonDescBuf;
        if(lessonDescBuf.size() != gDataset.size()){
            lessonDescBuf.resize(gDataset.size());
            for(size_t i=0;i<gDataset.size();++i){
                std::snprintf(lessonDescBuf[i].data(), lessonDescBuf[i].size(), "%s", gDataset[i].desc.c_str());
            }
        }
        int pendingDelete = -1;
        if(ImGui::BeginTable("lessons_table", 5, ImGuiTableFlags_Borders|ImGuiTableFlags_RowBg|ImGuiTableFlags_Resizable|ImGuiTableFlags_SizingStretchSame)){
            ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 40.0f);
            ImGui::TableSetupColumn("in");
            ImGui::TableSetupColumn("out");
            ImGui::TableSetupColumn("desc", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("actions", ImGuiTableColumnFlags_WidthFixed, 160.0f);
            ImGui::TableHeadersRow();
            int showN = (int)gDataset.size();
            for(int i=0;i<showN;i++){
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%d", i);
                auto& s = gDataset[i];
                auto vecToStr=[&](const std::vector<float>& v){ std::string r="["; for(size_t j=0;j<v.size();++j){ char b[32]; std::snprintf(b,sizeof(b),"%.2f", v[j]); r+=b; if(j+1<v.size()) r+=","; } r+="]"; return r; };
                ImGui::TableSetColumnIndex(1); { std::string ins = vecToStr(s.in); ImGui::TextUnformatted(ins.c_str()); }
                ImGui::TableSetColumnIndex(2); { std::string outs= vecToStr(s.out); ImGui::TextUnformatted(outs.c_str()); }
                ImGui::TableSetColumnIndex(3);
                ImGui::PushID(i);
                ImGui::SetNextItemWidth(-1);
                ImGui::InputText("##desc", lessonDescBuf[i].data(), lessonDescBuf[i].size());
                ImGui::PopID();
                ImGui::TableSetColumnIndex(4);
                ImGui::PushID(i+100000);
                if(ImGui::Button("Save")){
                    s.desc = std::string(lessonDescBuf[i].data());
                    Logf("Lesson %d description saved.", i);
                }
                ImGui::SameLine();
                if(ImGui::Button("Delete")){
                    pendingDelete = i;
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        if(pendingDelete >= 0 && pendingDelete < (int)gDataset.size()){
            gDataset.erase(gDataset.begin() + pendingDelete);
            gLessonsCount = (int)gDataset.size();
            if((int)lessonDescBuf.size() > pendingDelete) lessonDescBuf.erase(lessonDescBuf.begin() + pendingDelete);
            Logf("Lesson %d deleted.", pendingDelete);
        }
        ImGui::SameLine();
        if(ImGui::Button("Save network (S)")){
            std::error_code _ec; std::filesystem::create_directories("saved", _ec);
            // timestamped default name
            {
                std::time_t t = std::time(nullptr);
                std::tm tm{};
#if defined(_WIN32)
                localtime_s(&tm, &t);
#else
                localtime_r(&t, &tm);
#endif
                char buf[128];
                std::strftime(buf, sizeof(buf), "saved/network_%Y%m%d_%H%M%S.json", &tm);
                std::snprintf(gSaveName, sizeof(gSaveName), "%s", buf);
            }
            gShowSaveMenu = true;
        }
        ImGui::SameLine();
        if(ImGui::Button("Load network (O)")){
            std::error_code _ec; std::filesystem::create_directories("saved", _ec);
            gShowLoadMenu = true;
        }
            ImGui::End();
        }

        // Top-left: Controller info (F8 mode)
        if(topDown2D){
            ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.85f);
            static bool sShowController = true;
            ImGui::Begin("Controller [F8]", &sShowController, ImGuiWindowFlags_NoResize|ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoCollapse);
            int sp = (selectedPlayer>=0 && selectedPlayer<(int)players.size()) ? selectedPlayer : 0;
            const char* nm = (!players.empty()? (players[sp].name.empty()?"Player":players[sp].name.c_str()) : "None");
            ImGui::Text("Selected: %s (#%d)", nm, sp);
            // Input mapping table: slots == number of input neurons
            // Build sources
            std::vector<std::string> srcNames; std::vector<float> srcValues; BuildParamSources(*this, srcNames, srcValues);
            // Find input layer
            int inputLayerIdx = -1; for(size_t i=0;i<graph.layers.size();++i){ if(graph.layers[i].kind==LayerKind::Input){ inputLayerIdx=(int)i; break; } }
            if(inputLayerIdx>=0){
                const auto& L = graph.layers[inputLayerIdx];
                if((int)inputParamSelection.size() != (int)L.nodeIdx.size()) inputParamSelection.assign(L.nodeIdx.size(), 0);
                if(ImGui::BeginTable("inp_map", 4, ImGuiTableFlags_Borders|ImGuiTableFlags_RowBg|ImGuiTableFlags_SizingStretchSame)){
                    ImGui::TableSetupColumn("Slot", ImGuiTableColumnFlags_WidthFixed, 40.0f);
                    ImGui::TableSetupColumn("Neuron", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                    ImGui::TableSetupColumn("Mapping");
                    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                    ImGui::TableHeadersRow();
                    for(size_t i=0;i<L.nodeIdx.size();++i){
                        int nAbs = L.nodeIdx[i]; const char* nlabel = (nAbs>=0 && nAbs<(int)graph.nodes.size()? graph.nodes[nAbs].label.c_str() : "");
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0); ImGui::Text("%d", (int)i+1);
                        ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(nlabel);
                        ImGui::TableSetColumnIndex(2);
                        // Build combo items string
                        std::string comboItems; for(size_t k=0;k<srcNames.size();++k){ comboItems += srcNames[k]; comboItems.push_back('\0'); }
                        int sel = inputParamSelection[i]; if(sel<0 || sel>=(int)srcNames.size()) sel=0; if(ImGui::Combo((std::string("##m_")+std::to_string(i)).c_str(), &sel, comboItems.c_str())){ inputParamSelection[i]=sel; }
                        ImGui::TableSetColumnIndex(3); ImGui::Text("%.3f", (sel>=0 && sel<(int)srcValues.size()? srcValues[sel] : 0.0f));
                    }
                    ImGui::EndTable();
                }
            } else {
                ImGui::TextDisabled("No input layer in the network.");
            }
            ImGui::End();
        }

        // Right: Top-down manager panel (players, AI, bushes)
        if(topDown2D){
            ImGuiCond layCond2 = gForceRelayoutWindows ? ImGuiCond_Always : ImGuiCond_FirstUseEver;
            ImGui::SetNextWindowPos(ImVec2(disp.x - 360.0f, 10.0f), layCond2);
            ImGui::SetNextWindowSize(ImVec2(350.0f, 480.0f), layCond2);
            static bool sShowSimMap = true;
            if(ImGui::Begin("Simulator Map [F8]", &sShowSimMap, ImGuiWindowFlags_NoCollapse)){
                // Players list
                ImGui::Text("Players");
                ImGui::SameLine();
                if(ImGui::Button("Camera Player Focus")){
                    int sp = (selectedPlayer>=0 && selectedPlayer<(int)players.size())? selectedPlayer : 0;
                    if(!players.empty()){ topDownCenterTarget = players[sp].pos; cam.target = {players[sp].pos.x, players[sp].pos.y, players[sp].pos.z}; }
                }
                ImGui::SameLine();
                ImGui::Checkbox("Camera Follow", &cameraFollow);
                if(ImGui::Button("Add Blue Player")){
                    PlayerEntity p; p.pos={ (rand()%200-100)/50.0f, 0.0f, (rand()%200-100)/50.0f };
                    p.color={0.2f,0.55f,0.95f}; p.yaw=0.0f; p.aiIndex=-1; players.push_back(p);
                    selectedPlayer = (int)players.size()-1;
                }
                ImGui::SameLine();
                if(ImGui::Button("Add Bush")){
                    Bush b; b.pos={ (rand()%200-100)/50.0f, 0.0f, (rand()%200-100)/50.0f }; b.size={0.6f,0.6f}; bushes.push_back(b);
                }
                // Player selector
                if(selectedPlayer<0 || selectedPlayer>=(int)players.size()) selectedPlayer = !players.empty()?0:-1;
                if(selectedPlayer>=0){
                    ImGui::Text("Selected: #%d", selectedPlayer);
                    ImGui::SliderFloat("Yaw", &players[selectedPlayer].yaw, -3.1416f, 3.1416f);
                    ImGui::SliderFloat3("Pos", &players[selectedPlayer].pos.x, -10.0f, 10.0f);
                }
                // (Saved Networks section removed)
                // Shape placement UI
                ImGui::Text("Place Shape");
                static int shapeIdx = 0; // 0=Wedge,1=Box,2=Cylinder,3=Tree,4=Water
                const char* shapeItems[] = {"Wedge","Box","Cylinder","Tree","Water"};
                static float col[3] = {0.4f,0.8f,1.0f};
                static char nameBuf[64] = "Shape";
                static float placeY = 0.25f;
                static float placeScale[3] = {1.0f,1.0f,1.0f};
                static int treeCounter = 0; // Tree_0 for first
                static int waterCounter = 1; // Water_1 for first
                if(ImGui::Combo("Shape", &shapeIdx, shapeItems, 5)){
                    // Auto-apply presets on selection change
                    if(shapeIdx == 3){ // Tree preset
                        // Brown color and tall scale
                        col[0]=0.55f; col[1]=0.35f; col[2]=0.15f;
                        placeScale[0]=0.8f; placeScale[1]=3.0f; placeScale[2]=0.8f;
                        std::snprintf(nameBuf, sizeof(nameBuf), "Tree_%d", treeCounter);
                    } else if(shapeIdx == 4){ // Water preset
                        // Blue color and flat wide
                        col[0]=0.2f; col[1]=0.5f; col[2]=1.0f;
                        placeScale[0]=3.0f; placeScale[1]=0.05f; placeScale[2]=3.0f;
                        std::snprintf(nameBuf, sizeof(nameBuf), "Water_%d", waterCounter);
                    }
                }
                ImGui::ColorEdit3("Color", col, ImGuiColorEditFlags_NoInputs);
                static int sShapeCounter = 1;
                ImGui::InputText("Name", nameBuf, sizeof(nameBuf));
                ImGui::InputFloat("Height Y", &placeY, 0,0, "%.2f");
                ImGui::InputFloat3("Scale", placeScale, "%.2f");
                if(ImGui::Button("Place")){
                    // Place in front of selected player
                    int sp = (selectedPlayer>=0 && selectedPlayer<(int)players.size())? selectedPlayer : 0;
                    float yaw = (!players.empty()? players[sp].yaw : 0.0f);
                    Vec3 base = (!players.empty()? players[sp].pos : Vec3{0,0,0});
                    float c = std::cos(yaw), s = std::sin(yaw);
                    // forward vector (screen up): (-sin, 0, cos)
                    Vec3 forward = {-s, 0.0f, c};
                    float offset = 1.4f;
                    ShapeEntity se;
                    // Choose type and defaults
                    if(shapeIdx==0) se.type = ShapeType::Wedge;
                    else if(shapeIdx==1) se.type = ShapeType::Box;
                    else /* 2,3,4 */ se.type = ShapeType::Cylinder;
                    // Auto name and presets for special shapes
                    if(shapeIdx==3){ // Tree
                        char autoName[64]; std::snprintf(autoName, sizeof(autoName), "Tree_%d", treeCounter++); se.name=autoName;
                        se.color={0.55f,0.35f,0.15f}; se.scale={0.8f,3.0f,0.8f};
                    } else if(shapeIdx==4){ // Water
                        char autoName[64]; std::snprintf(autoName, sizeof(autoName), "Water_%d", waterCounter++); se.name=autoName;
                        se.color={0.2f,0.5f,1.0f}; se.scale={3.0f,0.05f,3.0f};
                    } else {
                        // Use user-provided values
                        se.name = nameBuf;
                        se.color = {col[0], col[1], col[2]}; se.scale = {placeScale[0], placeScale[1], placeScale[2]};
                    }
                    se.pos = { base.x + forward.x*offset, placeY, base.z + forward.z*offset };
                    se.yaw = yaw;
                    shapes.push_back(std::move(se));
                    // Update input list with Dist<Name> neuron
                    ShapeEntity& sref = shapes.back();
                    // Ensure input layer exists
                    int inputLayerIdx = -1; for(size_t i=0;i<graph.layers.size();++i){ if(graph.layers[i].kind==LayerKind::Input){ inputLayerIdx=(int)i; break; } }
                    if(inputLayerIdx<0){ Layer L; L.kind=LayerKind::Input; L.name="Inputs"; L.act=Act::Linear; graph.layers.insert(graph.layers.begin(), L); inputLayerIdx=0; }
                    NG_AddNeuron(graph, inputLayerIdx);
                    {
                        int absIdx = graph.layers[inputLayerIdx].nodeIdx.back();
                        if(absIdx>=0 && absIdx<(int)graph.nodes.size()){
                            std::string lab = std::string("Dist") + sref.name;
                            graph.nodes[absIdx].label = lab;
                            graph.nodes[absIdx].isInput = true;
                            sref.inputIdDist = graph.nodes[absIdx].id;
                        }
                    }
                    // Precompute next suggested name for generic shapes only
                    if(shapeIdx<=2){ const char* baseName = shapeItems[shapeIdx]; std::snprintf(nameBuf, sizeof(nameBuf), "%s%d", baseName, sShapeCounter++); }
                }
                ImGui::Separator();
                // Shapes table
                if(ImGui::BeginTable("shapes_tbl", 9, ImGuiTableFlags_Borders|ImGuiTableFlags_RowBg|ImGuiTableFlags_SizingStretchSame)){
                    ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 32.0f);
                    ImGui::TableSetupColumn("Name");
                    ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                    ImGui::TableSetupColumn("Color", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                    ImGui::TableSetupColumn("Pos", ImGuiTableColumnFlags_WidthFixed, 240.0f);
                    ImGui::TableSetupColumn("Yaw", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                    ImGui::TableSetupColumn("Scale", ImGuiTableColumnFlags_WidthFixed, 180.0f);
                    ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                    ImGui::TableHeadersRow();
                    int removeIdx = -1;
                    for(int i=0;i<(int)shapes.size();++i){
                        auto& se = shapes[i];
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0); ImGui::Text("%d", i);
                        ImGui::TableSetColumnIndex(1);
                        char nbuf[64]; std::snprintf(nbuf, sizeof(nbuf), "%s", se.name.c_str());
                        if(ImGui::InputText((std::string("##n")+std::to_string(i)).c_str(), nbuf, sizeof(nbuf))){
                            std::string old = se.name; se.name = nbuf;
                            if(se.inputIdDist!=-1){ EnsureCache(graph); auto it = graph.cache.id2idx.find(se.inputIdDist); if(it!=graph.cache.id2idx.end()){ std::string lab = std::string("Dist") + se.name; graph.nodes[it->second].label = lab; } }
                        }
                        ImGui::TableSetColumnIndex(2);
                        const char* tname = (se.type==ShapeType::Wedge?"Wedge": se.type==ShapeType::Box?"Box":"Cylinder");
                        ImGui::TextUnformatted(tname);
                        ImGui::TableSetColumnIndex(3);
                        float c3[3] = {se.color.x, se.color.y, se.color.z};
                        if(ImGui::ColorEdit3((std::string("##c")+std::to_string(i)).c_str(), c3, ImGuiColorEditFlags_NoInputs)){ se.color = {c3[0],c3[1],c3[2]}; }
                        ImGui::TableSetColumnIndex(4);
                        ImGui::PushItemWidth(60);
                        ImGui::InputFloat((std::string("X##")+std::to_string(i)).c_str(), &se.pos.x, 0,0,"%.2f"); ImGui::SameLine();
                        ImGui::InputFloat((std::string("Y##")+std::to_string(i)).c_str(), &se.pos.y, 0,0,"%.2f"); ImGui::SameLine();
                        ImGui::InputFloat((std::string("Z##")+std::to_string(i)).c_str(), &se.pos.z, 0,0,"%.2f");
                        ImGui::PopItemWidth();
                        ImGui::TableSetColumnIndex(5);
                        ImGui::SliderFloat((std::string("##yaw")+std::to_string(i)).c_str(), &se.yaw, -3.1416f, 3.1416f);
                        ImGui::TableSetColumnIndex(6);
                        ImGui::PushItemWidth(50);
                        ImGui::InputFloat((std::string("SX##")+std::to_string(i)).c_str(), &se.scale.x, 0,0,"%.2f"); ImGui::SameLine();
                        ImGui::InputFloat((std::string("SY##")+std::to_string(i)).c_str(), &se.scale.y, 0,0,"%.2f"); ImGui::SameLine();
                        ImGui::InputFloat((std::string("SZ##")+std::to_string(i)).c_str(), &se.scale.z, 0,0,"%.2f");
                        ImGui::PopItemWidth();
                        ImGui::TableSetColumnIndex(7);
                        if(ImGui::Button((std::string("Select##")+std::to_string(i)).c_str())){ selectedShape = i; topDownCenterTarget = se.pos; if(!topDown2D){ cam.target = {se.pos.x, se.pos.y, se.pos.z}; } }
                        ImGui::SameLine();
                        if(ImGui::Button((std::string("Delete##")+std::to_string(i)).c_str())) removeIdx = i;
                    }
                    ImGui::EndTable();
                    if(removeIdx>=0 && removeIdx<(int)shapes.size()) shapes.erase(shapes.begin()+removeIdx);
                }
                ImGui::End();
            }
        }
    }

    // Load menu (choose a network from 'saved')
    if(gShowLoadMenu){
        ImGui::SetNextWindowSize(ImVec2(520, 420), ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Load Network", &gShowLoadMenu, ImGuiWindowFlags_NoCollapse)){
            static std::vector<std::string> files;
            static int selected = -1;
            // Refresh list on open: simple heuristic
            if(ImGui::IsWindowAppearing()){
                files.clear(); selected = -1;
                std::error_code ec; std::filesystem::create_directories("saved", ec);
                for(const auto& de : std::filesystem::directory_iterator("saved", ec)){
                    if(!de.is_regular_file()) continue;
                    auto p = de.path();
                    if(p.extension() == ".json") files.push_back(p.filename().string());
                }
                std::sort(files.begin(), files.end());
            }

            if(ImGui::Button("Refresh")){
                files.clear(); selected = -1;
                std::error_code ec;
                for(const auto& de : std::filesystem::directory_iterator("saved", ec)){
                    if(!de.is_regular_file()) continue;
                    auto p = de.path();
                    if(p.extension() == ".json") files.push_back(p.filename().string());
                }
                std::sort(files.begin(), files.end());
            }
            ImGui::SameLine();
            if(ImGui::Button("Cancel")){
                gShowLoadMenu = false;
            }
            ImGui::Separator();

            ImGui::Text("Folder: %s", "saved/");
            ImVec2 listSize = ImVec2(-1, ImGui::GetTextLineHeightWithSpacing()*14.0f);
            if(ImGui::BeginListBox("##saved_list", listSize)){
                for(int i=0;i<(int)files.size();++i){
                    bool sel = (i==selected);
                    if(ImGui::Selectable(files[i].c_str(), sel)) selected = i;
                    if(sel) ImGui::SetItemDefaultFocus();
                    if(ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)){
                        // Double-click to load
                        std::string path = std::string("saved/") + files[i];
                        std::string msg;
                        if(NG_LoadJson(graph, path, &msg)){
                            gToastMsg = std::string(u8"Network loaded: ") + files[i];
                            Logf("Network loaded: %s", path.c_str());
                            gToastT = glfwGetTime();
                            gShowLoadMenu = false;
                        } else {
                            gToastMsg = std::string(u8"Load failed: ") + msg;
                            Logf("Load failed: %s", msg.c_str());
                            gToastT = glfwGetTime();
                        }
                    }
                }
                ImGui::EndListBox();
            }

            bool canLoad = selected >= 0 && selected < (int)files.size();
            if(!files.empty()) ImGui::Text("Selected: %s", canLoad ? files[selected].c_str() : "");
            if(ImGui::Button("Load") && canLoad){
                std::string path = std::string("saved/") + files[selected];
                std::string msg;
                if(NG_LoadJson(graph, path, &msg)){
                    gToastMsg = std::string(u8"Network loaded: ") + files[selected];
                    Logf("Network loaded: %s", path.c_str());
                    gToastT = glfwGetTime();
                    gShowLoadMenu = false;
                    if(gEngine) gEngine->networkDirty = false;
                } else {
                    gToastMsg = std::string(u8"Load failed: ") + msg;
                    Logf("Fuck! Load failed: %s", msg.c_str());
                    gToastT = glfwGetTime();
                }
            }
        }
        ImGui::End();
    }

    // Save menu (enter file name to save under 'saved/')
    if(gShowSaveMenu){
        ImGui::SetNextWindowSize(ImVec2(480, 140), ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Save Network", &gShowSaveMenu, ImGuiWindowFlags_NoCollapse)){
            ImGui::Text("Folder: saved/");
            ImGui::InputText("File name", gSaveName, IM_ARRAYSIZE(gSaveName));
            
            // Check for Enter key press to confirm save - only when input field is focused
            bool enterPressed = (ImGui::IsItemFocused() || ImGui::IsItemActive()) && 
                               (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter));
            bool shouldSave = false;
            
            ImGui::Spacing();
            if(ImGui::Button("Save")){
                shouldSave = true;
            }
            ImGui::SameLine();
            if(ImGui::Button("Cancel")) gShowSaveMenu = false;
            
            // Handle Enter key or Save button
            if(shouldSave || enterPressed){
                std::string path = gSaveName;
                // ensure .json
                if(path.size()<5 || path.substr(path.size()-5) != ".json") path += ".json";
                std::error_code ec; std::filesystem::create_directories("saved", ec);
                NG_SaveJson(graph, path);
                SaveEngineConf("engine.conf");
                gToastMsg = std::string(u8"Shitty Network saved: ") + path;
                gToastT = glfwGetTime();
                Logf("Total-shit Network saved: %s", path.c_str());
                gShowSaveMenu = false;
                if(gEngine) gEngine->networkDirty = false;
            }
        }
        ImGui::End();
    }

    // Exit confirmation modal
    if(gShowExitConfirm){
        ImGui::OpenPopup("Confirm Exit");
        if(ImGui::BeginPopupModal("Confirm Exit", &gShowExitConfirm, ImGuiWindowFlags_AlwaysAutoResize)){
            bool dirty = (gEngine && gEngine->networkDirty);
            ImGui::TextUnformatted(dirty ? "Unsaved changes detected in current network." : "");
            ImGui::Text("Exit the application?");
            ImGui::Separator();
            if(ImGui::Button("Exit", ImVec2(120,0))){
                glfwSetWindowShouldClose((GLFWwindow*)gEngine->window, 1);
                ImGui::CloseCurrentPopup();
                gShowExitConfirm = false;
            }
            ImGui::SameLine();
            if(ImGui::Button("Save and Exit", ImVec2(140,0))){
                // Quick timestamped save into saved/
                std::error_code _ec; std::filesystem::create_directories("saved", _ec);
                std::time_t t = std::time(nullptr);
                std::tm tm{};
#if defined(_WIN32)
                localtime_s(&tm, &t);
#else
                localtime_r(&t, &tm);
#endif
                char buf[128];
                std::strftime(buf, sizeof(buf), "saved/network_%Y%m%d_%H%M%S.json", &tm);
                NG_SaveJson(graph, buf);
                SaveEngineConf("engine.conf");
                if(gEngine) gEngine->networkDirty = false;
                glfwSetWindowShouldClose((GLFWwindow*)gEngine->window, 1);
                ImGui::CloseCurrentPopup();
                gShowExitConfirm = false;
            }
            ImGui::SameLine();
            if(ImGui::Button("Cancel", ImVec2(120,0))){ ImGui::CloseCurrentPopup(); gShowExitConfirm = false; }
            ImGui::EndPopup();
        }
    }

    // Help overlay (keybinds + quick start)
    if(gShowHelp){
        ImGui::SetNextWindowBgAlpha(0.92f);
        ImGui::SetNextWindowPos(ImVec2(18, 18), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.92f);
        ImVec2 helpCenter = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(helpCenter, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        
        ImGui::SetNextWindowBgAlpha(0.92f);

        ImGui::SetNextWindowPos(helpCenter, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::Begin("Help & Shortcuts [F5]", &gShowHelp,
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoSavedSettings);
        char help_title[128];
        std::snprintf(help_title, sizeof(help_title), "NeuralT - Help (%s)", kAppVersion);
        ImGui::TextColored(ImVec4(1,0.95f,0.6f,1), help_title);
        ImGui::Separator();
        if(ImGui::BeginTabBar("help_tabs", ImGuiTabBarFlags_AutoSelectNewTabs)){
            if(ImGui::BeginTabItem("Shortcuts & Quick Start")){
                ImGui::BulletText("H                    - toggle this help");
                ImGui::BulletText("F1                   - toggle Neural Editor");
                ImGui::BulletText("F2                   - NeuralT panel (visual settings)");
                ImGui::BulletText("F3                   - Log window");
                ImGui::BulletText("F4                   - Training parameters");
                ImGui::BulletText("F5                   - Help");
                ImGui::BulletText("F6                   - Live Feed window (inputs)");
                ImGui::BulletText("F8                   - toggle 2D top-down / 3D view");
                ImGui::BulletText("Top-down map: bottom-center Arrow 3 points at turbine");
                ImGui::BulletText("[ / ]                - previous / next layer");
                ImGui::BulletText("1..9                 - select layer by index");
                ImGui::BulletText("Ctrl+N               - new empty layers (1+3+1)");
                ImGui::BulletText("N / A / D            - add neuron / add hidden layer / delete last neuron in layer");
                ImGui::BulletText("S / O                - save / load network (open dialogs)");
                ImGui::BulletText("M                    - add a lesson from current IN/OUT");
                ImGui::BulletText("B / T                - train / train-to-target");
                ImGui::BulletText("W / Q / TAB          - live feed wave / randomize inputs / toggle all live");
                ImGui::BulletText("V / R / G / Shift+T  - VSync / rotate / schematic / theme");
                ImGui::Separator();
                ImGui::Text("Turbine");
                ImGui::BulletText("RPM slider range: 0..4000 (default 10)");
                ImGui::BulletText("Map arrow: Arrow 3 marks turbine position in top-down view");
                ImGui::Separator();
                ImGui::Text("Quick start");
                ImGui::BulletText("1) Create or load a network (Ctrl+N: empty 1+3+1 layers)");
                ImGui::BulletText("2) Add lessons (M) with current inputs/outputs");
                ImGui::BulletText("3) Tune epochs, learning rate, L1/L2, optional Target MSE");
                ImGui::BulletText("4) Start training (B); watch MSE trend");
                ImGui::BulletText("ESC - confirm exit (stops training if running)");
                ImGui::Separator();
                ImGui::EndTabItem();
            }
            if(ImGui::BeginTabItem("Neural Basics (PL)")){
                // Nowa, poprawiona wersja tekstu edukacyjnego (PL)
                ImGui::Text(u8"Czym jest neuron?");
                ImGui::TextWrapped(u8"Neuron oblicza wartość y = f(∑ w_i·x_i + b). Wejścia x_i mnożymy przez wagi w_i, dodajemy bias b i przepuszczamy przez funkcję aktywacji f.");
                ImGui::Separator();

                ImGui::Text(u8"Warstwy i topologia");
                ImGui::BulletText(u8"Warstwa wejściowa: podaje sygnały x (zwykle bez aktywacji)");
                ImGui::BulletText(u8"Warstwy ukryte: uczą się reprezentacji (np. ReLU, Tanh)");
                ImGui::BulletText(u8"Warstwa wyjściowa: zwraca wynik (np. Sigmoid dla przedziału 0..1)");
                ImGui::BulletText(u8"Więcej warstw/neuronów ⇒ większa pojemność, ale trudniejsze uczenie");
                ImGui::Separator();

                ImGui::Text(u8"Funkcje aktywacji");
                ImGui::BulletText(u8"Linear: f(x)=x (brak nieliniowości)");
                ImGui::BulletText(u8"ReLU: max(0,x) — szybka i stabilna dla głębokich sieci");
                ImGui::BulletText(u8"Sigmoid: 1/(1+e^-x) — wygodne wyjście 0..1");
                ImGui::BulletText(u8"Tanh: tanh(x) — wyjście −1..1");
                ImGui::Separator();

                ImGui::Text(u8"Propagacja w przód (forward)");
                ImGui::TextWrapped(u8"Dla każdej warstwy liczymy sumę wejść pomnożonych przez wagi + bias, po czym stosujemy aktywację. Tak sygnał przepływa od wejścia do wyjścia.");
                ImGui::Separator();

                ImGui::Text(u8"Uczenie i propagacja wstecz (backprop)");
                ImGui::TextWrapped(u8"Wybieramy funkcję straty (np. MSE). Liczymy błąd na wyjściu i propagujemy go wstecz, wyznaczając pochodne po wagach i biasach. Aktualizujemy parametry metodą spadku gradientowego (krok=learning rate), opcjonalnie z regularyzacją L1/L2 (ograniczanie przetrenowania).");
                ImGui::Separator();

                ImGui::Text(u8"Dane i normalizacja");
                ImGui::BulletText(u8"Skaluj wejścia do 0..1 lub −1..1 (łatwiejsze uczenie)");
                ImGui::BulletText(u8"Utrzymuj spójne jednostki i zakresy");
                ImGui::BulletText(u8"Mieszaj dane (shuffle), dziel na zbiory: trening/walidacja");
                ImGui::Separator();

                ImGui::Text(u8"Przykład intuicyjny");
                ImGui::TextWrapped(u8"Wejścia: x1=‘odległość’, x2=‘kąt’. Warstwa ukryta: 2 neurony ReLU. Wyjście: 1 neuron Sigmoid (0..1). Uczymy na parach (x1,x2)→t (0 lub 1). Po kilkudziesięciu epokach sieć uczy się reguły: im bliżej i ‘lepszy’ kąt, tym większa szansa=1.");
                ImGui::Separator();

                ImGui::Text(u8"Wskazówki praktyczne");
                ImGui::BulletText(u8"Zaczynaj od prostych topologii (np. 3-5-3)");
                ImGui::BulletText(u8"Jeśli MSE stoi: zmień lr/epoki, zmniejsz L2, sprawdź dane");
                ImGui::BulletText(u8"Monitoruj MSE na walidacji (wykres) — unikaj przeuczenia");
                ImGui::BulletText(u8"Zapisuj postęp i wersje sieci (łatwy powrót)");
                // old content below (was #if 1)
                ImGui::Text(u8"Czym jest neuron?");
                ImGui::TextWrapped(u8"Neuron oblicza wartość y = f(∑ w_i·x_i + b). \nWejścia x_i mnożymy przez wagi w_i, dodajemy bias b i przepuszczamy przez funkcję aktywacji f.");
                ImGui::Separator();

                ImGui::Text(u8"Warstwy i topologia");
                ImGui::BulletText(u8"Warstwa wejściowa: podaje sygnały x (zwykle bez aktywacji)");
                ImGui::BulletText(u8"Warstwy ukryte: uczą się reprezentacji (np. ReLU, Tanh)");
                ImGui::BulletText(u8"Warstwa wyjściowa: zwraca wynik (np. Sigmoid dla przedziału 0..1)");
                ImGui::BulletText(u8"Więcej warstw/neuronów ⇒ większa pojemność, ale trudniejsze uczenie");
                ImGui::Separator();

                ImGui::Text(u8"Funkcje aktywacji");
                ImGui::BulletText(u8"Linear: f(x)=x (brak nieliniowości)");
                ImGui::BulletText(u8"ReLU: max(0,x) — szybka i stabilna dla głębokich sieci");
                ImGui::BulletText(u8"Sigmoid: 1/(1+e^-x) — wygodne wyjście 0..1");
                ImGui::BulletText(u8"Tanh: tanh(x) — wyjście −1..1");
                ImGui::Separator();

                ImGui::Text(u8"Propagacja w przód (forward)");
                ImGui::TextWrapped(u8"Dla każdej warstwy liczymy sumę wejść pomnożonych przez wagi + bias,\npo czym stosujemy aktywację. Tak sygnał przepływa od wejścia do wyjścia.");
                ImGui::Separator();

                ImGui::Text(u8"Uczenie i propagacja wstecz (backprop)");
                ImGui::TextWrapped(u8"Wybieramy funkcję straty (np. MSE). Liczymy błąd na wyjściu i propagujemy go wstecz,\nwyznaczając pochodne po wagach i biasach. Aktualizujemy parametry metodą spadku gradientowego\n(krok=learning rate), opcjonalnie z regularyzacją L1/L2 (ograniczanie przetrenowania).");
                ImGui::Separator();

                ImGui::Text(u8"Dane i normalizacja");
                ImGui::BulletText(u8"Skaluj wejścia do 0..1 lub −1..1 (łatwiejsze uczenie)");
                ImGui::BulletText(u8"Utrzymuj spójne jednostki i zakresy");
                ImGui::BulletText(u8"Mieszaj dane (shuffle), dziel na zbiory: trening/walidacja");
                ImGui::Separator();

                ImGui::Text(u8"Przykład intuicyjny");
                ImGui::TextWrapped(u8"Wejścia: x1=‘odległość’, x2=‘kąt’. Warstwa ukryta: 2 neurony ReLU.\nWyjście: 1 neuron Sigmoid (0..1). Uczymy na parach (x1,x2)→t (0 lub 1).\nPo kilkudziesięciu epokach sieć uczy się reguły: im bliżej i ‘lepszy’ kąt, tym większa szansa=1.");
                ImGui::Separator();

                ImGui::Text(u8"Wskazówki praktyczne");
                ImGui::BulletText(u8"Zaczynaj od prostych topologii (np. 3-5-3)");
                ImGui::BulletText(u8"Jeśli MSE stoi: zmień lr/epoki, zmniejsz L2, sprawdź dane");
                ImGui::BulletText(u8"Monitoruj MSE na walidacji (wykres) — unikaj przeuczenia");
                ImGui::BulletText(u8"Zapisuj postęp i wersje sieci (łatwy powrót)");
                ImGui::TextWrapped(u8"Neuron: prosta funkcja obliczajÄ…ca wartoĹ›Ä‡ y = f(suma w_i * x_i + b). x_i to wejĹ›cia, w_i wagi, b to bias, f to funkcja aktywacji.");
                ImGui::Separator();
                ImGui::Text(u8"Warstwy i topologia");
                ImGui::BulletText(u8"Warstwa wejĹ›ciowa: dostarcza sygnaĹ‚y x (bez aktywacji)");
                ImGui::BulletText(u8"Warstwy ukryte: uczÄ… siÄ™ reprezentacji (np. ReLU/Tanh)");
                ImGui::BulletText(u8"Warstwa wyjĹ›ciowa: generuje wynik (np. Sigmoid dla 0..1)");
                ImGui::BulletText(u8"WiÄ™cej warstw i neuronĂłw => wiÄ™ksza pojemnoĹ›Ä‡, ale trudniejsze uczenie");
                ImGui::Separator();
                ImGui::Text(u8"Funkcje aktywacji");
                ImGui::BulletText(u8"Linear: f(x)=x (bez nieliniowoĹ›ci)");
                ImGui::BulletText(u8"ReLU: max(0,x) â€” szybka i stabilna dla gĹ‚Ä™bokich sieci");
                ImGui::BulletText(u8"Sigmoid: 1/(1+e^-x) â€” wyjĹ›cie 0..1");
                ImGui::BulletText(u8"Tanh: tanh(x) â€” wyjĹ›cie -1..1");
                ImGui::Separator();
                ImGui::Text(u8"Propagacja w przĂłd (forward)");
                ImGui::TextWrapped(u8"Dla kaĹĽdej warstwy ukrytej/wyjĹ›ciowej liczymy sumÄ™ wejĹ›Ä‡ pomnoĹĽonych przez wagi + bias, po czym stosujemy funkcjÄ™ aktywacji. Tak sygnaĹ‚ przepĹ‚ywa od wejĹ›cia do wyjĹ›cia.");
                ImGui::Separator();
                ImGui::Text(u8"Uczenie i propagacja wstecz (backprop)");
                ImGui::TextWrapped(u8"Definiujemy funkcjÄ™ straty (np. MSE). Liczymy bĹ‚Ä…d na wyjĹ›ciu, po czym propagujemy go wstecz warstwa po warstwie, wyznaczajÄ…c pochodne po wagach i biasach. Aktualizujemy parametry w kierunku zmniejszajÄ…cym stratÄ™ (gradient descent) z krokiem = learning rate, z opcjonalnym L1/L2.");
                ImGui::Separator();
                ImGui::Text(u8"PrzykĹ‚ad (intuicyjny)");
                ImGui::TextWrapped(u8"WejĹ›cia: x1=\"odlegĹ‚oĹ›Ä‡\", x2=\"gĹ‚Ăłd\". Warstwa ukryta: 2 neurony ReLU. WyjĹ›cie: 1 neuron Sigmoid (0..1). Uczymy na kilku lekcjach: [x1,x2] -> t (0 lub 1). Po kilkudziesiÄ™ciu epokach wyjĹ›cie przybliĹĽa reguĹ‚Ä™: im bliĹĽej i wiÄ™kszy gĹ‚Ăłd, tym wyĹĽsza szansa=1.");
                ImGui::Separator();
                ImGui::Text(u8"WskazĂłwki");
                ImGui::BulletText("Normalizuj dane do 0..1");
                ImGui::BulletText("Zaczynaj od prostych topologii (np. 3-5-3)");
                ImGui::BulletText(u8"JeĹ›li MSE stoi, zmieĹ„ lr/epoki lub zmniejsz L2");
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::End();
        }

    // Toast overlay
    if(!gToastMsg.empty()){
        double now = glfwGetTime();
        if(now - gToastT < 4.0){
            ImGui::SetNextWindowBgAlpha(0.92f);
            ImGui::SetNextWindowPos(ImVec2(18, 18), ImGuiCond_Always);
            ImGui::Begin("Message", nullptr, ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoSavedSettings);
            ImGui::TextUnformatted(gToastMsg.c_str());
            ImGui::End();
        }
    }

    // Training step & result banner (state is file-scope globals)
    if(gTrain.active || gShowTrainingWindow){
        // run one epoch per frame while active
        if(gTrain.active){
            double t0 = nowSeconds();
            gUseManualOverrides = false;
            float mse = TrainOneEpoch(graph, gTrain.lr, gTrain.l1, gTrain.l2);
            gUseManualOverrides = true;
            gTrain.lastMSE = mse;
            gTrain.mseHistory.push_back(mse);
            gTrain.epoch++;
            double t1 = nowSeconds();
            double dt = (t1 - t0);
            gTrain.avgEpochMs = (gTrain.avgEpochMs*0.9 + dt*1000.0*0.1);

            // Early stop on target MSE
            bool targetReached = (gTrain.targetMSE>0.0f && mse <= gTrain.targetMSE);

            if(gTrain.epoch >= gTrain.totalEpochs || targetReached){
                gTrain.active=false; gTrain.done=true;
                gShowTrainingWindow = false;
                gHUDHasRect = false;
                SaveEngineConf("engine.conf");
                NG_SaveJson(graph, "network_trained.json");
                Logf("Training finished: epoch=%d, MSE=%.6f%s.",
                     gTrain.epoch, mse, targetReached?" (target eliminated.)": "");
                gCelebrationVisible = true;
                gTrainDoneTime = nowSeconds();
                gTargetReachedVisual = targetReached;
                gTrainInterrupted = false;
            }
        }

        // draw HUD (MSE plot etc.) â€“ no screen dimming
        DrawTrainingHUD();
        // keep dim overlay: gHUDHasRect as set
        if(gHUDHasRect){
            ImDrawList* fg = ImGui::GetForegroundDrawList();
            ImVec2 sz = ImGui::GetIO().DisplaySize;
            ImU32 shade = ImGui::GetColorU32(ImVec4(0,0,0,0.6f));
            // top
            fg->AddRectFilled(ImVec2(0,0), ImVec2(sz.x, gHUDMin.y), shade);
            // left
            fg->AddRectFilled(ImVec2(0,gHUDMin.y), ImVec2(gHUDMin.x, gHUDMax.y), shade);
            // right
            fg->AddRectFilled(ImVec2(gHUDMax.x, gHUDMin.y), ImVec2(sz.x, gHUDMax.y), shade);
            // bottom
            fg->AddRectFilled(ImVec2(0, gHUDMax.y), ImVec2(sz.x, sz.y), shade);
        }

        if(gTrain.done && gCelebrationVisible){
                // Centered result window with very large font
                ImVec2 winSz(720, 460);
                ImVec2 sz = ImGui::GetIO().DisplaySize;
                ImGui::SetNextWindowPos(ImVec2((sz.x-winSz.x)*0.5f, (sz.y-winSz.y)*0.5f), ImGuiCond_Always);
                ImGui::SetNextWindowSize(winSz, ImGuiCond_Always);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(18,16));
                ImVec4 bg = gTargetReachedVisual ? ImVec4(0.12f,0.55f,0.22f,0.95f) : ImVec4(0.90f,0.55f,0.15f,0.95f);
                ImGui::PushStyleColor(ImGuiCol_WindowBg, bg);
                ImGui::Begin("TrainingResult", nullptr, ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoInputs);
                // Title (always "Training Finished") and status line
                ImGui::SetWindowFontScale(3.0f);
                const char* titleTop = "Training Finished";
                ImVec2 ts = ImGui::CalcTextSize(titleTop);
                ImGui::SetCursorPosX((winSz.x - ts.x)*0.5f);
                ImGui::TextUnformatted(titleTop);
                ImGui::SetWindowFontScale(1.4f);
                const char* status = gTrainInterrupted ? "Training Interrupted" : (gTargetReachedVisual ? "Target Reached" : "Target Not Reached");
                ImVec2 st = ImGui::CalcTextSize(status);
                ImGui::SetCursorPosX((winSz.x - st.x)*0.5f);
                ImGui::TextUnformatted(status);
                ImGui::SetWindowFontScale(1.0f);
                ImGui::Spacing(); ImGui::Separator();
                // IQ gain (heuristic)
                float mse0 = (!gTrain.mseHistory.empty()? gTrain.mseHistory.front() : gTrain.lastMSE);
                float improve = std::max(0.0f, mse0 - gTrain.lastMSE);
                float ratio = (mse0>1e-6f) ? (improve / mse0) : 0.0f;
                int iqGain = (int)std::round(std::max(0.0f, ratio) * 100.0f);
                ImGui::Spacing();
                ImGui::SetWindowFontScale(2.0f);
                int currentIQ = 100 + iqGain;
                char iqbuf[64]; std::snprintf(iqbuf, sizeof(iqbuf), "Current IQ: %d", currentIQ);
                ImVec2 iqts = ImGui::CalcTextSize(iqbuf);
                ImGui::SetCursorPosX((winSz.x - iqts.x)*0.5f);
                ImGui::TextUnformatted(iqbuf);
                ImGui::SetWindowFontScale(1.0f);
                ImGui::Spacing(); ImGui::Separator();
                // Summary
                ImGui::Text("Epoch: %d / %d", gTrain.epoch, gTrain.totalEpochs);
                ImGui::Text("Final MSE: %.6f  Target: %s", gTrain.lastMSE, gTrain.targetMSE>0?std::to_string(gTrain.targetMSE).c_str():"-");
                ImGui::Spacing();
                // Final MSE plot with axes and legend
                ImVec2 plotSize(std::max(400.0f, winSz.x - 80.0f), 160.0f);
                ImGui::PlotLines("##final_mse", gTrain.mseHistory.data(), (int)gTrain.mseHistory.size(), 0, nullptr, FLT_MAX, FLT_MAX, plotSize);
                {
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    ImVec2 p0 = ImGui::GetItemRectMin();
                    ImVec2 p1 = ImGui::GetItemRectMax();
                    // Axes
                    dl->AddLine(ImVec2(p0.x, p0.y), ImVec2(p0.x, p1.y), ImGui::GetColorU32(ImVec4(0.85f,0.85f,0.9f,0.8f)), 1.0f);
                    dl->AddLine(ImVec2(p0.x, p1.y), ImVec2(p1.x, p1.y), ImGui::GetColorU32(ImVec4(0.85f,0.85f,0.9f,0.8f)), 1.0f);
                    // Ticks + labels
                    const int xt = 8, yt = 4;
                    for(int i=0;i<=xt;i++){
                        float t=(float)i/(float)xt; float x=p0.x + t*(p1.x-p0.x);
                        dl->AddLine(ImVec2(x,p1.y), ImVec2(x,p1.y-4.0f), ImGui::GetColorU32(ImVec4(0.8f,0.8f,0.9f,0.7f)), 1.0f);
                    }
                    // y-range estimation
                    float ymin=1e9f, ymax=-1e9f; int n=(int)gTrain.mseHistory.size();
                    for(int i=0;i<n;i++){ float v=gTrain.mseHistory[i]; if(v<ymin) ymin=v; if(v>ymax) ymax=v; }
                    if(!(ymax>ymin)){ ymin=0.0f; ymax=1.0f; }
                    for(int i=0;i<=yt;i++){
                        float t=(float)i/(float)yt; float y=p1.y - t*(p1.y-p0.y);
                        dl->AddLine(ImVec2(p0.x,y), ImVec2(p0.x+5.0f,y), ImGui::GetColorU32(ImVec4(0.8f,0.8f,0.9f,0.7f)), 1.0f);
                        float v=ymin + t*(ymax-ymin); char buf[32]; std::snprintf(buf,sizeof(buf),"%.3f", v);
                        dl->AddText(ImVec2(p0.x+6.0f, y-8.0f), ImGui::GetColorU32(ImVec4(0.85f,0.9f,1.0f,0.9f)), buf);
                    }
                    // Target line
                    if(gTrain.targetMSE>0.0f){
                        float ty = p1.y - (gTrain.targetMSE - ymin) / std::max(1e-6f, (ymax-ymin)) * (p1.y-p0.y);
                        dl->AddLine(ImVec2(p0.x,ty), ImVec2(p1.x,ty), ImGui::GetColorU32(ImVec4(0.95f,0.4f,0.2f,0.9f)), 1.5f);
                    }
                    // Legend (inside plot, top-left)
                    ImVec2 lp = ImVec2(p0.x + 6.0f, p0.y + 6.0f);
                    dl->AddRectFilled(ImVec2(lp.x, lp.y), ImVec2(lp.x+10, lp.y+10), ImGui::GetColorU32(ImVec4(0.2f,0.85f,0.7f,1.0f)));
                    dl->AddText(ImVec2(lp.x+14, lp.y-2), ImGui::GetColorU32(ImVec4(0.9f,0.95f,1,1)), "MSE");
                    if(gTrain.targetMSE>0.0f){
                        float ox = lp.x + 60.0f;
                        dl->AddRectFilled(ImVec2(ox, lp.y), ImVec2(ox+10, lp.y+10), ImGui::GetColorU32(ImVec4(0.95f,0.4f,0.2f,1.0f)));
                        dl->AddText(ImVec2(ox+14, lp.y-2), ImGui::GetColorU32(ImVec4(0.9f,0.95f,1,1)), "Target");
                    }
                }
                ImGui::End();
                ImGui::PopStyleColor();
                ImGui::PopStyleVar();
        }
    }

    /* Stirlitz przyszedĹ‚ na tajne spotkanie pod umĂłwiony adres o umĂłwionej godzinie i minucie. 
       ZapukaĹ‚ umĂłwieone 286 razy, ale nikt nie otworzyĹ‚. Stirlitz wychyliĹ‚ siÄ™ i spojrzaĹ‚ na parapet. 
       StaĹ‚o tam 7 ĹĽelazek - znak wpadki. */

    // Draw result window even when HUD is hidden or training inactive
#ifdef USE_IMGUI
    if(gCelebrationVisible && !(gTrain.active || gShowTrainingWindow)){
        ImVec2 winSz(720, 460);
        ImVec2 sz = ImGui::GetIO().DisplaySize;
        ImGui::SetNextWindowPos(ImVec2((sz.x-winSz.x)*0.5f, (sz.y-winSz.y)*0.5f), ImGuiCond_Always);
        ImGui::SetNextWindowSize(winSz, ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(18,16));
        ImVec4 bg = gTargetReachedVisual ? ImVec4(0.12f,0.55f,0.22f,0.95f) : ImVec4(0.90f,0.55f,0.15f,0.95f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, bg);
        ImGui::Begin("TrainingResult", nullptr, ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoInputs);
        // Title (always "Training Finished") and status line
        ImGui::SetWindowFontScale(3.0f);
        const char* titleTop = "Training Finished";
        ImVec2 ts = ImGui::CalcTextSize(titleTop);
        ImGui::SetCursorPosX((winSz.x - ts.x)*0.5f);
        ImGui::TextUnformatted(titleTop);
        ImGui::SetWindowFontScale(1.4f);
        const char* status = gTrainInterrupted ? "EMERGENCY PUSSY-BUTTON PRESSED!!!" : (gTargetReachedVisual ? "Target Reached" : "Target Not Reached");
        ImVec2 st = ImGui::CalcTextSize(status);
        ImGui::SetCursorPosX((winSz.x - st.x)*0.5f);
        ImGui::TextUnformatted(status);
        ImGui::SetWindowFontScale(1.0f);
        ImGui::Spacing(); ImGui::Separator();
        // IQ gain (heuristic)
        float mse0 = (!gTrain.mseHistory.empty()? gTrain.mseHistory.front() : gTrain.lastMSE);
        float improve = std::max(0.0f, mse0 - gTrain.lastMSE);
        float ratio = (mse0>1e-6f) ? (improve / mse0) : 0.0f;
        int iqGain = (int)std::round(std::max(0.0f, ratio) * 100.0f);
        ImGui::Spacing();
        ImGui::SetWindowFontScale(2.0f);
        int currentIQ = 100 + iqGain;
        char iqbuf[64]; std::snprintf(iqbuf, sizeof(iqbuf), "Current IQ: %d", currentIQ);
        ImVec2 iqts = ImGui::CalcTextSize(iqbuf);
        ImGui::SetCursorPosX((winSz.x - iqts.x)*0.5f);
        ImGui::TextUnformatted(iqbuf);
        ImGui::SetWindowFontScale(1.0f);
        ImGui::Spacing(); ImGui::Separator();
        // Summary
        ImGui::Text("Epoch: %d / %d", gTrain.epoch, gTrain.totalEpochs);
        ImGui::Text("Final MSE: %.6f  Target: %s", gTrain.lastMSE, gTrain.targetMSE>0?std::to_string(gTrain.targetMSE).c_str():"-");
        ImGui::Spacing();
        // Final MSE plot with axes and legend
        ImVec2 plotSize(std::max(400.0f, winSz.x - 80.0f), 160.0f);
        ImGui::PlotLines("##final_mse2", gTrain.mseHistory.data(), (int)gTrain.mseHistory.size(), 0, nullptr, FLT_MAX, FLT_MAX, plotSize);
        {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            ImVec2 p0 = ImGui::GetItemRectMin();
            ImVec2 p1 = ImGui::GetItemRectMax();
            // Axes
            dl->AddLine(ImVec2(p0.x, p0.y), ImVec2(p0.x, p1.y), ImGui::GetColorU32(ImVec4(0.85f,0.85f,0.9f,0.8f)), 1.0f);
            dl->AddLine(ImVec2(p0.x, p1.y), ImVec2(p1.x, p1.y), ImGui::GetColorU32(ImVec4(0.85f,0.85f,0.9f,0.8f)), 1.0f);
            // Ticks + labels
            const int xt = 8, yt = 4;
            for(int i=0;i<=xt;i++){
                float t=(float)i/(float)xt; float x=p0.x + t*(p1.x-p0.x);
                dl->AddLine(ImVec2(x,p1.y), ImVec2(x,p1.y-4.0f), ImGui::GetColorU32(ImVec4(0.8f,0.8f,0.9f,0.7f)), 1.0f);
            }
            // y-range estimation
            float ymin=1e9f, ymax=-1e9f; int n=(int)gTrain.mseHistory.size();
            for(int i=0;i<n;i++){ float v=gTrain.mseHistory[i]; if(v<ymin) ymin=v; if(v>ymax) ymax=v; }
            if(!(ymax>ymin)){ ymin=0.0f; ymax=1.0f; }
            for(int i=0;i<=yt;i++){
                float t=(float)i/(float)yt; float y=p1.y - t*(p1.y-p0.y);
                dl->AddLine(ImVec2(p0.x,y), ImVec2(p0.x+5.0f,y), ImGui::GetColorU32(ImVec4(0.8f,0.8f,0.9f,0.7f)), 1.0f);
                float v=ymin + t*(ymax-ymin); char buf[32]; std::snprintf(buf,sizeof(buf),"%.3f", v);
                dl->AddText(ImVec2(p0.x+6.0f, y-8.0f), ImGui::GetColorU32(ImVec4(0.85f,0.9f,1.0f,0.9f)), buf);
            }
            // Target line
            if(gTrain.targetMSE>0.0f){
                float ty = p1.y - (gTrain.targetMSE - ymin) / std::max(1e-6f, (ymax-ymin)) * (p1.y-p0.y);
                dl->AddLine(ImVec2(p0.x,ty), ImVec2(p1.x,ty), ImGui::GetColorU32(ImVec4(0.95f,0.4f,0.2f,0.9f)), 1.5f);
            }
            // Legend (inside plot, top-left)
            ImVec2 lp = ImVec2(p0.x + 6.0f, p0.y + 6.0f);
            dl->AddRectFilled(ImVec2(lp.x, lp.y), ImVec2(lp.x+10, lp.y+10), ImGui::GetColorU32(ImVec4(0.2f,0.85f,0.7f,1.0f)));
            dl->AddText(ImVec2(lp.x+14, lp.y-2), ImGui::GetColorU32(ImVec4(0.9f,0.95f,1,1)), "MSE");
            if(gTrain.targetMSE>0.0f){
                float ox = lp.x + 60.0f;
                dl->AddRectFilled(ImVec2(ox, lp.y), ImVec2(ox+10, lp.y+10), ImGui::GetColorU32(ImVec4(0.95f,0.4f,0.2f,1.0f)));
                dl->AddText(ImVec2(ox+14, lp.y-2), ImGui::GetColorU32(ImVec4(0.9f,0.95f,1,1)), "Target");
            }
        }
        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
    }
#endif

    // Top-down map arrow ("Arrow 3"): draw at bottom-center pointing to turbine (screen center)
// (Removed old 2D overlay arrow; now drawn as 3D mesh in world)

    // After layout pass, clear the relayout flag
    if(gForceRelayoutWindows) gForceRelayoutWindows = false;
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
#endif

    {
        static int lastSwapInterval = -1;
        int desired = cfg.vsync ? 1 : 0;
        if(desired != lastSwapInterval){ glfwSwapInterval(desired); lastSwapInterval = desired; }
    }
    glfwSwapBuffers((GLFWwindow*)window);
    glfwPollEvents();
}

void Engine::shutdown(){
#ifdef USE_IMGUI
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
#endif
    renderer.destroy();
    turbine.destroy();
    if(window){ glfwDestroyWindow((GLFWwindow*)window); window=nullptr; }
    glfwTerminate();
}

void Engine::onResize(int /*w*/, int /*h*/){}

static InputState gInput;

void Engine::onMouseButton(int button, int action, int mods){
    if(button==0) gInput.mouseDownL = (action==1);
    if(button==1) gInput.mouseDownR = (action==1);
    if(button==2) gInput.mouseDownM = (action==1);
    gInput.alt = (mods & 0x0004) != 0;
    gInput.ctrl = (mods & 0x0002) != 0;
    gInput.shift= (mods & 0x0001) != 0;

#ifdef USE_IMGUI
    // Close celebration banner on any click
    if(action==1){
// celebration handled elsewhere (no-op)
        // We intentionally rely on frame logic to hide it after input via onKey below.
    }
#endif

    // Selection in top-down mode with left click: pick nearest shape, then player
    if(topDown2D && button==0 && action==1){
#ifdef USE_IMGUI
        if(ImGui::GetIO().WantCaptureMouse) return;
#endif
        int w,h; glfwGetFramebufferSize((GLFWwindow*)window, &w, &h);
        float aspect = (h>0)? (float)w/(float)h : 1.0f;
        float base = 6.0f; float left,right,bottom,top;
        if(aspect>=1.0f){ top=base; bottom=-base; right=base*aspect; left=-right; }
        else { right=base; left=-base; top=base/aspect; bottom=-top; }
        // Convert cursor (pixels) to world X,Z (ortho)
        float mx = (float)gInput.mouseX; float my = (float)gInput.mouseY;
        float wx = left + (mx/(float)w) * (right-left);
        float wz = top  - (my/(float)h) * (top-bottom);
        // Find nearest shape first
        int bestShape=-1; float bestSD=1e9f;
        for(int i=0;i<(int)shapes.size();++i){ float dx=shapes[i].pos.x-wx; float dz=shapes[i].pos.z-wz; float d=dx*dx+dz*dz; if(d<bestSD){ bestSD=d; bestShape=i; } }
        if(bestShape>=0 && bestSD < (1.0f*1.0f)){
            selectedShape = bestShape; selectedPlayer = -1;
            topDownCenterTarget = shapes[bestShape].pos;
            Logf("Selected shape: %s (#%d)", shapes[bestShape].name.c_str(), bestShape);
            return;
        }
        // Then nearest player
        int best=-1; float bestD=1e9f;
        for(int i=0;i<(int)players.size();++i){ float dx=players[i].pos.x-wx; float dz=players[i].pos.z-wz; float d=dx*dx+dz*dz; if(d<bestD){ bestD=d; best=i; } }
        if(best>=0 && bestD < (0.9f*0.9f)){
            selectedPlayer = best; selectedShape = -1;
            topDownCenterTarget = players[best].pos;
            Logf("Selected controller: %s (index %d)", players[best].name.empty()?"Player":players[best].name.c_str(), best);
        }
    }
}
void Engine::onCursorPos(double x, double y){
    double dx = x - gInput.mouseX;
    double dy = y - gInput.mouseY;
    gInput.mouseX = x; 
    gInput.mouseY = y;

#ifdef USE_IMGUI
    if(ImGui::GetIO().WantCaptureMouse) return;
#endif
    if(gInput.mouseDownL && !gInput.shift) cam.orbit((float)dx, (float)dy);
    if(gInput.mouseDownM || (gInput.mouseDownL && gInput.shift)) cam.pan((float)dx,(float)dy);
}
void Engine::onScroll(double /*dx*/, double dy){
#ifdef USE_IMGUI
    if(ImGui::GetIO().WantCaptureMouse) return;
#endif
    cam.dolly((float)dy);
}
// Update players in F8 mode: player[0] manual like a boat (only forward, arc turns), others optional AI
void Engine::updateTopDownPlayers(float dt){
    GLFWwindow* win = (GLFWwindow*)window;
    if(!win) return;
    const float turnSpeed = 1.8f; // rad/s
    const float moveSpeed = 2.5f; // world units per second

    // Manual control for selected player (default 0)
    if(!players.empty()){
        int sp = (selectedPlayer>=0 && selectedPlayer<(int)players.size()) ? selectedPlayer : 0;
        float turn = 0.0f; float move = 0.0f;
        // Normal mapping: LEFT = turn left (negative yaw), RIGHT = turn right (positive yaw)
        if(glfwGetKey(win, GLFW_KEY_LEFT) == GLFW_PRESS)  turn -= 1.0f;
        if(glfwGetKey(win, GLFW_KEY_RIGHT)== GLFW_PRESS)  turn += 1.0f;
        // Forward only in the direction currently facing
        if(glfwGetKey(win, GLFW_KEY_UP) == GLFW_PRESS)  move = 1.0f; // only forward, no reverse
        players[sp].yaw += turn * turnSpeed * dt;
        if(move > 0.0f){
            float c = std::cos(players[sp].yaw), s = std::sin(players[sp].yaw);
            // Move in the exact facing direction on screen: right turn -> top-right path
            players[sp].pos.x -= s * (moveSpeed * move * dt);
            players[sp].pos.z += c * (moveSpeed * move * dt);
            const float kHalf = 10.0f;
            players[sp].pos.x = std::max(-kHalf, std::min(kHalf, players[sp].pos.x));
            players[sp].pos.z = std::max(-kHalf, std::min(kHalf, players[sp].pos.z));
        }
    }

    // AI control for players (except #0 manual) and all creatures
    auto norm01 = [](float v){ const float kHalf=10.0f; float t=0.5f+0.5f*(v/kHalf); return std::max(0.0f,std::min(1.0f,t)); };
    for(size_t pi=0; pi<players.size(); ++pi){ if((int)pi==0) continue; auto& p = players[pi];
        if(p.aiIndex<0 || p.aiIndex>=(int)aiControllers.size()) continue;
        auto& ai = aiControllers[p.aiIndex]; if(!ai.loaded) continue;
        // Fill inputs: if model expects >=22 inputs, use predator feature vector; else fallback to PosX/Y/Z
        size_t inCount = ai.inputAbsIdx.size();
        if(inCount >= 22){
            // Choose prey: default players[0] if exists (not self), else next player or origin
            Vec3 preyPos = {0,0,0};
            if(!players.empty()){
                if(pi!=0) preyPos = players[0].pos; else if(players.size()>1) preyPos = players[1].pos;
            }
            auto clampHalf = [&](float v){ const float kHalf=10.0f; return std::max(-kHalf, std::min(kHalf, v)); };
            auto prox = [&](float dist){ const float full=20.0f; float d = std::max(0.0f, std::min(full, dist)); return 1.0f - (d/full); };
            float yawS = std::sin(p.yaw), yawC = std::cos(p.yaw);
            float tdx = clampHalf(preyPos.x - p.pos.x), tdz = clampHalf(preyPos.z - p.pos.z);
            float tdist = std::sqrt(tdx*tdx + tdz*tdz);
            float odx = clampHalf(-p.pos.x), odz = clampHalf(-p.pos.z);
            float odist = std::sqrt(odx*odx + odz*odz);
            // find two nearest bushes
            int b1=-1, b2=-1; float d1=1e9f, d2=1e9f;
            for(size_t bi=0; bi<bushes.size(); ++bi){ float dx = bushes[bi].pos.x - p.pos.x; float dz = bushes[bi].pos.z - p.pos.z; float d=dx*dx+dz*dz; if(d<d1){ d2=d1;b2=b1; d1=d; b1=(int)bi; } else if(d<d2){ d2=d; b2=(int)bi; } }
            float b1dx=0,b1dz=0,b1dist=0,b2dx=0,b2dz=0,b2dist=0;
            if(b1>=0){ b1dx = clampHalf(bushes[b1].pos.x - p.pos.x); b1dz = clampHalf(bushes[b1].pos.z - p.pos.z); b1dist = std::sqrt(b1dx*b1dx + b1dz*b1dz); }
            if(b2>=0){ b2dx = clampHalf(bushes[b2].pos.x - p.pos.x); b2dz = clampHalf(bushes[b2].pos.z - p.pos.z); b2dist = std::sqrt(b2dx*b2dx + b2dz*b2dz); }

            float feat[22] = {
                norm01(p.pos.x), norm01(p.pos.y), norm01(p.pos.z),
                (yawS+1.0f)*0.5f, (yawC+1.0f)*0.5f,
                norm01(tdx), norm01(tdz), std::min(1.0f, tdist/ (2*10.0f)),
                norm01(odx), norm01(odz), std::min(1.0f, odist/ (2*10.0f)),
                norm01(b1dx), norm01(b1dz), std::min(1.0f, b1dist/ (2*10.0f)),
                norm01(b2dx), norm01(b2dz), std::min(1.0f, b2dist/ (2*10.0f)),
                // wall proximity (1 near wall)
                1.0f - std::min(1.0f, (p.pos.x + 10.0f) / 20.0f),
                1.0f - std::min(1.0f, (10.0f - p.pos.x) / 20.0f),
                1.0f - std::min(1.0f, (10.0f - p.pos.z) / 20.0f),
                1.0f - std::min(1.0f, (p.pos.z + 10.0f) / 20.0f),
                (rand()/(float)RAND_MAX)
            };
            for(size_t i=0;i<22 && i<ai.inputAbsIdx.size(); ++i){ int abs=ai.inputAbsIdx[i]; if(abs>=0 && abs<(int)ai.net.nodes.size()) ai.net.nodes[abs].value = feat[i]; }
        } else {
            // Fallback: PosX/Y/Z only
            for(size_t i=0;i<inCount && i<ai.inputAbsIdx.size(); ++i){
                int abs = ai.inputAbsIdx[i]; if(abs<0 || abs>=(int)ai.net.nodes.size()) continue;
                float val = 0.0f; if(i==0) val = norm01(p.pos.x); else if(i==1) val = norm01(p.pos.y); else if(i==2) val = norm01(p.pos.z);
                ai.net.nodes[abs].value = val;
            }
        }
        // Forward
        ForwardPass(ai.net);
        // Read outputs
        float forward = 0.0f, turnLR = 0.0f, boost=0.0f; // turnLR -1..1
        if(ai.outputAbsIdx.size()>=3){
            forward = ai.net.nodes[ai.outputAbsIdx[0]].value; // 0..1
            turnLR = ai.net.nodes[ai.outputAbsIdx[1]].value * 2.0f - 1.0f; // map 0..1 -> -1..1
            boost   = ai.net.nodes[ai.outputAbsIdx[2]].value; // 0..1
        } else if(ai.outputAbsIdx.size()>=2){
            forward = ai.net.nodes[ai.outputAbsIdx[0]].value;
            turnLR = ai.net.nodes[ai.outputAbsIdx[1]].value * 2.0f - 1.0f;
        } else if(ai.outputAbsIdx.size()==1){
            forward = (ai.net.nodes[ai.outputAbsIdx[0]].value>0.5f)?1.0f:0.0f;
            turnLR = 0.0f;
        }
        p.yaw += turnLR * turnSpeed * dt;
        float speedMul = 0.6f + 1.4f * std::min(1.0f, std::max(0.0f, boost));
        if(forward>0.01f){
            float c = std::cos(p.yaw), s = std::sin(p.yaw);
            // Move along mesh forward (+Z rotated by yaw): (x,z) += (sin(yaw), cos(yaw))
            p.pos.x += s * (moveSpeed * speedMul * forward * dt);
            p.pos.z += c * (moveSpeed * speedMul * forward * dt);
            const float kHalf = 10.0f;
            p.pos.x = std::max(-kHalf, std::min(kHalf, p.pos.x));
            p.pos.z = std::max(-kHalf, std::min(kHalf, p.pos.z));
        }
    }
}

// Create/ensure 3 input neurons bound to player position
void Engine::ensurePlayerInputs(){
    // If already set, nothing to do
    if(playerInputIdX!=-1 && playerInputIdY!=-1 && playerInputIdZ!=-1) return;
    int inputLayerIdx = -1;
    for(size_t i=0;i<graph.layers.size();++i){ if(graph.layers[i].kind==LayerKind::Input){ inputLayerIdx=(int)i; break; } }
    if(inputLayerIdx<0){
        Layer L; L.kind=LayerKind::Input; L.name="Inputs"; L.act=Act::Linear;
        graph.layers.insert(graph.layers.begin(), L);
        inputLayerIdx = 0;
    }
    // Add 3 neurons and label them
    auto addOne = [&](const char* label)->int{
        int before = (int)graph.nodes.size();
        NG_AddNeuron(graph, inputLayerIdx);
        int absIdx = graph.layers[inputLayerIdx].nodeIdx.back();
        if(absIdx >= 0 && absIdx < (int)graph.nodes.size()){
            graph.nodes[absIdx].label = label;
            graph.nodes[absIdx].isInput = true;
            return graph.nodes[absIdx].id;
        }
        // fallback: return last id change
        return (before<(int)graph.nodes.size()) ? graph.nodes.back().id : -1;
    };
    if(playerInputIdX==-1) playerInputIdX = addOne("PosX");
    if(playerInputIdY==-1) playerInputIdY = addOne("PosY");
    if(playerInputIdZ==-1) playerInputIdZ = addOne("PosZ");
    networkDirty = true;
    Logf("Player inputs added: PosX=%d PosY=%d PosZ=%d", playerInputIdX, playerInputIdY, playerInputIdZ);
}

void Engine::ensurePlayerYawInput(){
    if(playerInputIdYaw != -1) return;
    int inputLayerIdx = -1;
    for(size_t i=0;i<graph.layers.size();++i){ if(graph.layers[i].kind==LayerKind::Input){ inputLayerIdx=(int)i; break; } }
    if(inputLayerIdx<0){ Layer L; L.kind=LayerKind::Input; L.name="Inputs"; L.act=Act::Linear; graph.layers.insert(graph.layers.begin(), L); inputLayerIdx=0; }
    NG_AddNeuron(graph, inputLayerIdx);
    int absIdx = graph.layers[inputLayerIdx].nodeIdx.back();
    if(absIdx >=0 && absIdx < (int)graph.nodes.size()){
        graph.nodes[absIdx].label = "Yaw";
        graph.nodes[absIdx].isInput = true;
        playerInputIdYaw = graph.nodes[absIdx].id;
    }
    networkDirty = true;
}

// Push current player coords into bound input neurons each frame
void Engine::pushPlayerCoordsToInputs(){
    auto setById = [&](int id, float v){ if(id==-1) return; 
        for(auto& n : graph.nodes){ if(n.id==id){ n.value = v; break; } }
    };
    // Normalize world coords to 0..1 based on map half-extent
    const float kHalf = 10.0f; // matches grid (N=20,S=0.5 => extent ~10)
    auto norm01 = [&](float v){ float t = 0.5f + 0.5f * (v / kHalf); return std::max(0.0f, std::min(1.0f, t)); };
    int sp = (selectedPlayer>=0 && selectedPlayer<(int)players.size()) ? selectedPlayer : 0;
    const Vec3 pos0 = (!players.empty()? players[sp].pos : Vec3{0,0,0});
    setById(playerInputIdX, norm01(pos0.x));
    setById(playerInputIdY, norm01(pos0.y));
    setById(playerInputIdZ, norm01(pos0.z));
    // Yaw normalized to 0..1 (-pi..pi -> 0..1)
    float yaw = (!players.empty()? players[sp].yaw : 0.0f);
    auto wrapPi = [&](float a){ while(a> 3.14159265f) a-=6.28318531f; while(a<-3.14159265f) a+=6.28318531f; return a; };
    float yaw01 = (wrapPi(yaw) + 3.14159265f) / (2.0f*3.14159265f);
    setById(playerInputIdYaw, yaw01);
    // Update distances for shape inputs
    updateShapeDistanceInputs();
}

void Engine::updateShapeDistanceInputs(){
    const float kHalf = 10.0f;
    auto normDist01 = [&](float d){ return std::max(0.0f, std::min(1.0f, d / (2.0f*kHalf))); };
    int sp = (selectedPlayer>=0 && selectedPlayer<(int)players.size()) ? selectedPlayer : 0;
    Vec3 p = (!players.empty()? players[sp].pos : Vec3{0,0,0});
    for(const auto& se : shapes){ if(se.inputIdDist==-1) continue; float dx = se.pos.x - p.x; float dz = se.pos.z - p.z; float d = std::sqrt(dx*dx+dz*dz); float v = normDist01(d);
        for(auto& n : graph.nodes){ if(n.id == se.inputIdDist){ n.value = v; break; } }
    }
}

// Build current parameter sources: names and values
static void BuildParamSources(const Engine& E, std::vector<std::string>& names, std::vector<float>& values){
    names.clear(); values.clear();
    const float kHalf = 10.0f;
    auto norm01 = [&](float v){ float t = 0.5f + 0.5f * (v / kHalf); return std::max(0.0f, std::min(1.0f, t)); };
    auto wrapPi = [&](float a){ while(a> 3.14159265f) a-=6.28318531f; while(a<-3.14159265f) a+=6.28318531f; return a; };
    int sp = (E.selectedPlayer>=0 && E.selectedPlayer<(int)E.players.size())? E.selectedPlayer : 0;
    Vec3 p = (!E.players.empty()? E.players[sp].pos : Vec3{0,0,0});
    float yaw = (!E.players.empty()? E.players[sp].yaw : 0.0f);
    // Core params
    names.push_back("PosX"); values.push_back(norm01(p.x));
    names.push_back("PosY"); values.push_back(norm01(p.y));
    names.push_back("PosZ"); values.push_back(norm01(p.z));
    names.push_back("Yaw");  values.push_back((wrapPi(yaw)+3.14159265f)/(2.0f*3.14159265f));
    // Dist<Name> from shapes
    for(const auto& se : E.shapes){
        float dx = se.pos.x - p.x; float dz = se.pos.z - p.z; float d = std::sqrt(dx*dx+dz*dz);
        std::string lab = std::string("Dist") + se.name;
        names.push_back(lab);
        values.push_back(std::max(0.0f, std::min(1.0f, d / (2.0f*kHalf))));
    }
}

void Engine::applyInputMappings(){
    // Build sources
    std::vector<std::string> srcNames; std::vector<float> srcValues;
    BuildParamSources(*this, srcNames, srcValues);
    // Find input layer
    int inputLayerIdx = -1; for(size_t i=0;i<graph.layers.size();++i){ if(graph.layers[i].kind==LayerKind::Input){ inputLayerIdx=(int)i; break; } }
    if(inputLayerIdx<0) return;
    const auto& L = graph.layers[inputLayerIdx];
    // Ensure selection size
    if((int)inputParamSelection.size() != (int)L.nodeIdx.size()) inputParamSelection.assign(L.nodeIdx.size(), 0);
    // Apply values per slot
    for(size_t i=0;i<L.nodeIdx.size();++i){ int nodeAbs = L.nodeIdx[i]; if(nodeAbs<0 || nodeAbs>=(int)graph.nodes.size()) continue; int sel = inputParamSelection[i]; if(sel<0 || sel>=(int)srcValues.size()) sel=0; graph.nodes[nodeAbs].value = srcValues[sel]; }
}

// Removed saved-network features
/*void Engine::assignAIToPlayer(int playerIdx, int aiIdx){
    if(playerIdx<0 || playerIdx>=(int)players.size()) return;
    if(aiIdx<0 || aiIdx>=(int)aiControllers.size()) return;
    players[playerIdx].aiIndex = aiIdx;
    Logf("Assigned %s to player #%d", aiControllers[aiIdx].name.c_str(), playerIdx);
}*/

// ---------------- Predator synthetic dataset and training -----------------
static void PredatorBuildFeatures(const Engine::PlayerEntity& p, const Vec3& preyPos, const std::vector<Engine::Bush>& bushes, float feat[22]){
    auto norm01 = [](float v){ const float kHalf=10.0f; float t=0.5f+0.5f*(v/kHalf); return std::max(0.0f,std::min(1.0f,t)); };
    auto clampHalf = [&](float v){ const float kHalf=10.0f; return std::max(-kHalf, std::min(kHalf, v)); };
    auto dist01 = [&](float dx, float dz){ float d = std::sqrt(dx*dx+dz*dz); return std::min(1.0f, d/(2*10.0f)); };
    float yawS = std::sin(p.yaw), yawC = std::cos(p.yaw);
    float tdx = clampHalf(preyPos.x - p.pos.x), tdz = clampHalf(preyPos.z - p.pos.z);
    float odx = clampHalf(-p.pos.x), odz = clampHalf(-p.pos.z);
    // two nearest bushes
    int b1=-1,b2=-1; float d1=1e9f,d2=1e9f;
    for(size_t bi=0; bi<bushes.size(); ++bi){ float dx=bushes[bi].pos.x-p.pos.x, dz=bushes[bi].pos.z-p.pos.z; float d=dx*dx+dz*dz; if(d<d1){ d2=d1;b2=b1; d1=d;b1=(int)bi; } else if(d<d2){ d2=d;b2=(int)bi; } }
    float b1dx=0,b1dz=0,b2dx=0,b2dz=0; float b1d=0,b2d=0;
    if(b1>=0){ b1dx = clampHalf(bushes[b1].pos.x-p.pos.x); b1dz = clampHalf(bushes[b1].pos.z-p.pos.z); b1d = std::sqrt(b1dx*b1dx+b1dz*b1dz);}    
    if(b2>=0){ b2dx = clampHalf(bushes[b2].pos.x-p.pos.x); b2dz = clampHalf(bushes[b2].pos.z-p.pos.z); b2d = std::sqrt(b2dx*b2dx+b2dz*b2dz);}    
    feat[0]=norm01(p.pos.x); feat[1]=norm01(p.pos.y); feat[2]=norm01(p.pos.z);
    feat[3]=(yawS+1.0f)*0.5f; feat[4]=(yawC+1.0f)*0.5f;
    feat[5]=norm01(tdx); feat[6]=norm01(tdz); feat[7]=dist01(tdx,tdz);
    feat[8]=norm01(odx); feat[9]=norm01(odz); feat[10]=dist01(odx,odz);
    feat[11]=norm01(b1dx); feat[12]=norm01(b1dz); feat[13]=std::min(1.0f,b1d/(2*10.0f));
    feat[14]=norm01(b2dx); feat[15]=norm01(b2dz); feat[16]=std::min(1.0f,b2d/(2*10.0f));
    feat[17]=1.0f - std::min(1.0f,(p.pos.x + 10.0f)/20.0f); // left wall proximity
    feat[18]=1.0f - std::min(1.0f,(10.0f - p.pos.x)/20.0f); // right
    feat[19]=1.0f - std::min(1.0f,(10.0f - p.pos.z)/20.0f); // top
    feat[20]=1.0f - std::min(1.0f,(p.pos.z + 10.0f)/20.0f); // bottom
    feat[21]=(rand()/(float)RAND_MAX);
}

static float wrapAngle(float a){ while(a> 3.14159265f) a-=6.28318531f; while(a<-3.14159265f) a+=6.28318531f; return a; }

static void PredatorComputeTargets(const Engine::PlayerEntity& p, const Vec3& preyPos, const std::vector<Engine::Bush>& bushes, float out15[15]){
    for(int i=0;i<15;++i) out15[i]=0.0f;
    // Turn command towards prey
    float angToPrey = std::atan2(preyPos.x - p.pos.x, preyPos.z - p.pos.z); // note X,Z
    float dAng = wrapAngle(angToPrey - p.yaw);
    float turnLR = std::max(-1.0f, std::min(1.0f, dAng / 1.2f)); // clamp
    out15[1] = 0.5f + 0.5f * turnLR; // Turn 0..1
    // Distances
    float dx = preyPos.x - p.pos.x, dz = preyPos.z - p.pos.z; float dist = std::sqrt(dx*dx+dz*dz);
    float dist01 = std::min(1.0f, dist / 10.0f);
    // Default forward based on distance (further -> more forward)
    out15[0] = std::min(1.0f, 0.2f + 0.9f * dist01);
    // If close enough -> attack (fast, boost)
    if(dist < 3.0f){ out15[3]=1.0f; out15[2]=1.0f; out15[0]=1.0f; }
    else if(dist < 6.0f){ out15[4]=1.0f; out15[2]=0.6f; }
    else { out15[14]=1.0f; }
    // Avoid nearest bush if it sits roughly between predator and prey
    int nearest=-1; float nd=1e9f;
    for(size_t i=0;i<bushes.size();++i){ float bx=bushes[i].pos.x-p.pos.x, bz=bushes[i].pos.z-p.pos.z; float d=bx*bx+bz*bz; if(d<nd){ nd=d; nearest=(int)i; } }
    if(nearest>=0){
        Vec3 bp = bushes[nearest].pos; // simple line check: if bush lies near segment predator-prey
        float t = ((bp.x - p.pos.x)*dx + (bp.z - p.pos.z)*dz) / std::max(1e-4f, dx*dx+dz*dz);
        t = std::max(0.0f, std::min(1.0f, t));
        float lx = p.pos.x + t*dx, lz = p.pos.z + t*dz;
        float off = std::sqrt((bp.x-lx)*(bp.x-lx)+(bp.z-lz)*(bp.z-lz));
        if(off < 0.9f){ out15[9]=1.0f; out15[0]=std::max(0.3f, out15[0]); }
    }
    // If near wall, set evade
    if(std::abs(p.pos.x)>9.0f || std::abs(p.pos.z)>9.0f){ out15[6]=1.0f; out15[2]=std::max(out15[2], 0.7f); }
    // Bias towards circling when at medium distance
    if(dist>=3.0f && dist<7.5f){ if(turnLR<0) out15[7]=1.0f; else out15[8]=1.0f; }
    // Guard center sometimes when very close to origin
    if(std::sqrt(p.pos.x*p.pos.x + p.pos.z*p.pos.z) < 2.0f) out15[11]=1.0f;
}

static void PredatorGenerateDataset(size_t N, std::vector<TrainSample>& outDs){
    outDs.clear(); outDs.reserve(N);
    Engine::PlayerEntity p{}; Engine::PlayerEntity prey{}; std::vector<Engine::Bush> bushes;
    for(size_t i=0;i<N;++i){
        // Randomize scenario
        p.pos = { (rand()%200-100)/10.0f, 0.0f, (rand()%200-100)/10.0f };
        p.yaw = (rand()/(float)RAND_MAX) * 6.28318531f - 3.14159265f;
        prey.pos = { (rand()%200-100)/10.0f, 0.0f, (rand()%200-100)/10.0f };
        bushes.clear(); int nb = 2 + (rand()%5);
        for(int b=0;b<nb;++b){ Engine::Bush bu; bu.pos = { (rand()%200-100)/10.0f, 0.0f, (rand()%200-100)/10.0f }; bu.size={0.6f,0.6f}; bushes.push_back(bu);}        
        float in[22]; float out[15]; PredatorBuildFeatures(p, prey.pos, bushes, in); PredatorComputeTargets(p, prey.pos, bushes, out);
        TrainSample s; s.in.assign(in, in+22); s.out.assign(out, out+15); s.desc = "predator"; outDs.push_back(std::move(s));
    }
}

static void PredatorTrainAI(NeuralGraph& net, const std::vector<TrainSample>& ds, int epochs, float lr, float l1, float l2){
    // Minimal copy of TrainOneEpoch that uses provided dataset
    auto TrainOneEpochWith = [&](NeuralGraph& g)->float{
        if(ds.empty()) return FLT_MAX; float mse=0.0f; EnsureCache(g);
        for(const auto& s : ds){
            size_t pi=0; for(const auto& L : g.layers){ if(L.kind!=LayerKind::Input) continue; for(int idx : L.nodeIdx){ g.nodes[idx].value = (pi < s.in.size() ? s.in[pi++] : 0.0f); }}
            ForwardPass(g);
            std::vector<int> outIdx; Act outAct=Act::Linear; for(const auto& L : g.layers){ if(L.kind==LayerKind::Output){ outAct=L.act; for(int idx: L.nodeIdx) outIdx.push_back(idx);} }
            std::vector<float> delta(g.nodes.size(), 0.0f);
            for(size_t i=0;i<outIdx.size();++i){ int idx=outIdx[i]; Neuron& n=g.nodes[idx]; float y=n.value; float t=(i<s.out.size()? s.out[i] : 0.0f); float err=y-t; mse += err*err; float d = err * ActDeriv(outAct, y); delta[idx]=d; }
            // Hidden layers backprop
            for(int li=(int)g.layers.size()-2; li>=0; --li){ const auto& L=g.layers[li]; if(L.kind==LayerKind::Output) continue; for(int idxN : L.nodeIdx){ int id=g.nodes[idxN].id; float sum=0.0f; auto it=g.cache.outEdges.find(id); if(it!=g.cache.outEdges.end()){
                            for(int ei: it->second){ const auto& e=g.edges[ei]; auto itd=g.cache.id2idx.find(e.b); if(itd!=g.cache.id2idx.end()){ sum += delta[itd->second]*e.w; }} }
                    float y=g.nodes[idxN].value; float d = sum * ActDeriv(L.act, y); delta[idxN]=d; }
            }
            // Update weights
            for(auto& n : g.nodes){ float b = ng_get_bias(n); b -= lr * (delta[g.cache.id2idx[n.id]] + l2*b + l1*sgn(b)); ng_set_bias(n, b); }
            for(auto& e : g.edges){ auto ita=g.cache.id2idx.find(e.a); if(ita!=g.cache.id2idx.end()){ float x=g.nodes[ita->second].value; e.w -= lr * (delta[g.cache.id2idx[e.b]]*x + l2*e.w + l1*sgn(e.w)); } }
        }
        return mse / (float)std::max<size_t>(1, ds.size());
    };
    for(int ep=0; ep<epochs; ++ep){ float mse = TrainOneEpochWith(net); if(ep%50==0) Logf("Predator train epoch %d MSE=%.6f", ep, mse); }
}

static void SetupPredatorExampleGraph(){
    // Build 22-31-38-33-26-15 network for predator scenario and prepare lessons
    if(!gEngine) return;
    auto& graph = gEngine->graph;
    NG_BuildTopology(graph, 22, std::vector<int>{31,38,33,26}, 15);
    // Label inputs for consistent mapping
    static const char* kInLabels[22] = {
        "PosX","PosY","PosZ","YawSin","YawCos",
        "TargetDX","TargetDZ","TargetDist",
        "TurbineDX","TurbineDZ","TurbineDist",
        "Bush1DX","Bush1DZ","Bush1Dist",
        "Bush2DX","Bush2DZ","Bush2Dist",
        "WallLeft","WallRight","WallTop","WallBottom",
        "Noise"
    };
    if(!graph.layers.empty() && graph.layers.front().kind==LayerKind::Input){
        auto& Lin = graph.layers.front();
        for(size_t i=0;i<Lin.nodeIdx.size() && i<22;i++){ graph.nodes[Lin.nodeIdx[i]].label = kInLabels[i]; }
    }
    if(!graph.layers.empty()){
        auto& Lout = graph.layers.back();
        if(Lout.kind==LayerKind::Output){
            for(size_t i=0;i<Lout.nodeIdx.size(); ++i){ int idx=Lout.nodeIdx[i];
                std::string l = (i==0?"Fwd": i==1?"Turn": i==2?"Boost": std::string("O")+std::to_string((int)i));
                graph.nodes[idx].label = l;
            }
        }
    }
    NG_RandomizeHidden(graph, 1.0f, 0.5f);
    // Generate synthetic predator dataset and attach to editor
    std::vector<TrainSample> ds; PredatorGenerateDataset(1200, ds);
    gDataset = ds; gLessonsCount = (int)gDataset.size();
    // Reasonable training defaults
    gTrain.lr = 0.01f; gTrain.l1 = 0.0001f; gTrain.l2 = 0.0001f; gTrain.totalEpochs = 1500; gTrain.targetMSE = 0.05f;
    SaveEngineConf("engine.conf");
    Logf("Predator Example initialized: topology 22-31-38-33-26-15, lessons=%d", gLessonsCount);
}
void Engine::onKey(int key, int /*scancode*/, int action, int mods){
    bool down = action != 0; if(!down) return;
    // If result window is visible, allow closing it with Enter/Space/Esc
#ifdef USE_IMGUI
    if(gCelebrationVisible){
        if(key==GLFW_KEY_ESCAPE || key==GLFW_KEY_ENTER || key==GLFW_KEY_KP_ENTER || key==GLFW_KEY_SPACE){
            gCelebrationVisible = false; gTrain.done = false; return;
        }
    }
#endif
    // ESC should interrupt training at any moment
    if(key==GLFW_KEY_ESCAPE){
        if(gTrain.active){
            gTrain.active = false;
            gShowTrainingWindow = false;
            gHUDHasRect = false;
            gTrain.done = true;
            gCelebrationVisible = true;
            gTargetReachedVisual = false;
            gTrainInterrupted = true;
            SaveEngineConf("engine.conf");
            Logf("Training interrupted (ESC) at epoch %d, MSE=%.6f", gTrain.epoch, gTrain.lastMSE);
            return;
        } else {
            // If not training, show exit confirmation (with unsaved info)
            gShowExitConfirm = true;
            return;
        }
    }
#ifdef USE_IMGUI
    // When editing text (e.g., labels) - don't run shortcuts
    ImGuiIO& _io = ImGui::GetIO();
    if(gEditingLabel || _io.WantTextInput || _io.WantCaptureKeyboard) return;
#endif
#ifdef USE_IMGUI
    // Function-key visibility toggles (blocked while editing text)
    if(key==GLFW_KEY_F1){ gShowEditorWindow = !gShowEditorWindow; return; }
    if(key==GLFW_KEY_F2){ gShowNeuralXPanel = !gShowNeuralXPanel; return; }
    if(key==GLFW_KEY_F3){ gShowLogWindow    = !gShowLogWindow; return; }
    if(key==GLFW_KEY_F4){ gShowParamsWindow = !gShowParamsWindow; return; }
    if(key==GLFW_KEY_F5){ gShowHelp = !gShowHelp; return; }
    if(key==GLFW_KEY_F6){ gShowLiveFeedWindow = !gShowLiveFeedWindow; return; }
    if(key==GLFW_KEY_F8){
        topDown2D = !topDown2D;
        if(topDown2D){
            // Reset selected player to bottom-center within ortho frame
            if(players.empty()) players.push_back({});
            int sp = (selectedPlayer>=0 && selectedPlayer<(int)players.size()) ? selectedPlayer : 0;
            players[sp].pos = {0.0f, 0.25f, -5.5f};
            players[sp].yaw = 0.0f;
            topDownCenter = players[sp].pos;
        }
        Logf(topDown2D?"View: Top-down 2D (F8)":"View: 3D perspective (F8)");
        return;
    }
    if(key==GLFW_KEY_F12) {
        graph.nodes.clear();
        graph.edges.clear();
        graph.layers.clear();
        graph.selected = -1;
        graph.currentLayer = -1; // Set to -1 to indicate no layer is selected
        gDataset.clear();
        gLessonsCount = 0;
        Logf("Cleared all neurons, layers, and lessons.");
        return;
    }
    if(key==GLFW_KEY_H){ gShowHelp = !gShowHelp; return; }

    if(key==GLFW_KEY_LEFT_BRACKET){
        graph.currentLayer = std::max(0, graph.currentLayer-1);
        return;
    }
    if(key==GLFW_KEY_RIGHT_BRACKET){
        graph.currentLayer = std::min((int)graph.layers.size()-1, graph.currentLayer+1);
        return;
    }
    // ------------------------------------------------------------------
    // New shortcut: Left Shift + 1
    // Assign ascending weights to all edges for color-gradient visualization
    // ------------------------------------------------------------------
    if(key==GLFW_KEY_1 && (mods & GLFW_MOD_SHIFT)){
        NG_AssignSortedWeights(graph);
        Logf("Assigned ascending weights (-1..1) across %zu edges.", graph.edges.size());
        return;
    }
    if(key>=GLFW_KEY_1 && key<=GLFW_KEY_9){
        int idx = key - GLFW_KEY_1; // 0-based
        if(idx < (int)graph.layers.size()) graph.currentLayer = idx;
        return;
    }
    if((mods & GLFW_MOD_CONTROL) && key==GLFW_KEY_N){
        // New empty layers (Ctrl+N): input + 3 hidden + output, no neurons
        gDataset.clear(); gLessonsCount = 0;
        gManualOutputs.clear(); gUseManualOverrides=false;
        NG_BuildTopology(graph, /*inN*/0, std::vector<int>{0,0,0}, /*outN*/0);
        graph.currentLayer = 0;
        if(gEngine) gEngine->networkDirty = true;
        Logf("New empty network created (layers only).");
        return;
    }
    if(key==GLFW_KEY_N){ NG_AddNeuron(graph, graph.currentLayer); return; }
    if(key==GLFW_KEY_D){ NG_RemoveLastNeuron(graph, graph.currentLayer); return; }
    if(key==GLFW_KEY_A){ NG_InsertHiddenAfterCurrent(graph); return; }
    if(key==GLFW_KEY_TAB){
        static bool s_allInputsLive = true; // toggled by TAB
        s_allInputsLive = !s_allInputsLive;
        graph.liveInputs = s_allInputsLive;
        // Apply to all input neurons
        for(const auto& L : graph.layers){
            if(L.kind != LayerKind::Input) continue;
            for(int idx : L.nodeIdx){
                if(idx<0 || idx>=(int)graph.nodes.size()) continue;
                auto& n = graph.nodes[idx];
                n.live = s_allInputsLive;
                if(s_allInputsLive && n.speed<=0.0f){ n.speed = 0.6f; n.phase = 0.0f; }
            }
        }
        return;
    }
    // 'S' handled below with save dialog (opens Save window)
    if(key==GLFW_KEY_X){ NG_RandomizeHidden(graph, 1.0f, 0.5f); Logf("Hidden layers randomized."); return; }
    if(key==GLFW_KEY_M){ std::vector<float> in,out; CollectIO(graph,in,out); gDataset.push_back({in,out}); gLessonsCount=(int)gDataset.size(); SaveEngineConf("engine.conf"); Logf("Lesson added. Total=%d", gLessonsCount); gManualOutputs.clear(); gUseManualOverrides=false; return; }
    if(key==GLFW_KEY_B){
        gShowTrainingWindow = true;
        if(!gDataset.empty()){
            // Break symmetry if all weights are equal (freshly built graphs)
            bool uniform = true; float ref = (!graph.edges.empty()? graph.edges.front().w : 0.0f);
            for(const auto& e_ : graph.edges){ if(std::fabs(e_.w - ref) > 1e-6f){ uniform=false; break; } }
            if(uniform) NG_RandomizeHidden(graph, 1.0f, 0.5f);
            gTrain.active=true; gTrain.done=false; gTrain.epoch=0; gTrain.mseHistory.clear(); gTrain.startTime=nowSeconds(); gTrain.avgEpochMs=0.0;
            Logf("Training started (hotkey B): epochs=%d, lr=%.6f, L1=%.6f, L2=%.6f, targetMSE=%.6f",
                 gTrain.totalEpochs, gTrain.lr, gTrain.l1, gTrain.l2, gTrain.targetMSE);
        } else {
            Logf("Cannot start training: dataset is empty.");
        }
        return;
    }
#endif
    if(key==GLFW_KEY_R){ cfg.rotate = !cfg.rotate; }
    if(key==GLFW_KEY_G){ cfg.showSchematic = !cfg.showSchematic; }
    if(key==GLFW_KEY_V){ cfg.vsync = !cfg.vsync; }
    if(key==GLFW_KEY_T){
        if((mods & GLFW_MOD_SHIFT) != 0){
                if(renderer.theme.name=="NeonNoir") renderer.theme = UITheme::DeepSpace();
            else if(renderer.theme.name=="DeepSpace") renderer.theme = UITheme::SolarFlare();
            else if(renderer.theme.name=="SolarFlare") renderer.theme = UITheme::Sunset();
            else if(renderer.theme.name=="Sunset") renderer.theme = UITheme::EmberForge();
            else if(renderer.theme.name=="EmberForge") renderer.theme = UITheme::Cyberpunk();
            else if(renderer.theme.name=="Cyberpunk") renderer.theme = UITheme::Crimson();
            else if(renderer.theme.name=="Crimson") renderer.theme = UITheme::Ruby();
            else if(renderer.theme.name=="Ruby") renderer.theme = UITheme::Aurora();
            else if(renderer.theme.name=="Aurora") renderer.theme = UITheme::ForestMist();
            else if(renderer.theme.name=="ForestMist") renderer.theme = UITheme::OceanWave();
            else if(renderer.theme.name=="OceanWave") renderer.theme = UITheme::MonoSlate();
            else /* Ruby or other */ renderer.theme = UITheme::NeonNoir();
            Logf("Theme selected: %s", renderer.theme.name.c_str());
            return;
        } else {
            // Train to target (fine-tune) via keyboard 't'
            if(!gDataset.empty() && gTrain.targetMSE > 0.0f){
                bool uniform = true; float ref = (!graph.edges.empty()? graph.edges.front().w : 0.0f);
                for(const auto& e_ : graph.edges){ if(std::fabs(e_.w - ref) > 1e-6f){ uniform=false; break; } }
                if(uniform) NG_RandomizeHidden(graph, 1.0f, 0.5f);
                gTrain.totalEpochs = std::max(gTrain.totalEpochs, 200000);
                gTrain.active=true; gTrain.done=false; gTrain.epoch=0; gTrain.mseHistory.clear(); gTrain.startTime = nowSeconds(); gTrain.avgEpochMs=0.0;
                Logf("Train-to-kill-target started (hotkey T): targetMSE=%.6f, epochs=%d, lr=%.6f", gTrain.targetMSE, gTrain.totalEpochs, gTrain.lr);
            } else {
                if(gDataset.empty()) Logf("Cannot train to target: dataset is empty you idiot.");
                else Logf("Cannot kill the target: set Target MSE > 0 first. Pussy.");
            }
            return;
        }
    }
    if(key==GLFW_KEY_SPACE){
        // Toggle live feed on selected input neuron
        int selId = graph.selected;
        if(selId != -1){
            for(auto& n : graph.nodes){
                if(n.id==selId && n.isInput){
                    bool was = n.live; n.live = !n.live;
                    if(n.live && !was && n.speed<=0.0f){ n.speed=0.6f; n.phase=0.0f; }
                    break;
                }
            }
        }
        return;
    }
    if(key==GLFW_KEY_S){
#ifdef USE_IMGUI
        std::error_code _ec; std::filesystem::create_directories("saved", _ec);
        // timestamped default name for quick save
        {
            std::time_t t = std::time(nullptr);
            std::tm tm{};
#if defined(_WIN32)
            localtime_s(&tm, &t);
#else
            localtime_r(&t, &tm);
#endif
            char buf[128];
            std::strftime(buf, sizeof(buf), "saved/network_%Y%m%d_%H%M%S.json", &tm);
            std::snprintf(gSaveName, sizeof(gSaveName), "%s", buf);
        }
        gShowSaveMenu = true;
        return;
#else
        NG_SaveJson(graph, "network.json"); 
        SaveEngineConf("engine.conf"); 
        Logf("Network saved: network.json"); 
        return;
#endif
    }
    if(key==GLFW_KEY_HOME){
        // Show all primary windows and re-layout to fit screen
        cfg.showUI = true;
        gShowEditorWindow = true;
        gShowNeuralXPanel = true;
        gShowLogWindow = true;
        gShowParamsWindow = true;
        gForceRelayoutWindows = true;
        return;
    }
    if(key==GLFW_KEY_O){
#ifdef USE_IMGUI
        std::error_code _ec; std::filesystem::create_directories("saved", _ec);
        gShowLoadMenu = true;
        return;
#else
        std::string msg; 
        if(NG_LoadJson(graph, "network.json", &msg))
        {
             Logf("Network loaded: network.json"); 
        } else 
        { 
            Logf("Load failed: %s", msg.c_str()); 
        }
        return;
#endif
    }
    if(key==GLFW_KEY_W){
        gLiveFeedWaveActive = !gLiveFeedWaveActive;
        if(gLiveFeedWaveActive){
            gLiveFeedCurrentInput = 0;
            gLiveFeedInputTimer = 0.0f;
            gLiveFeedWavePhase = 0.0f;
            Logf("Sekwencyjne sinusoidy dla wejĹ›Ä‡ uruchomione (W)");
        } else {
            Logf("Sekwencyjne sinusoidy dla wejĹ›Ä‡ wyĹ‚Ä…czone (W)");
        }
        return;
    }
    if(key==GLFW_KEY_Q){
        // Random live feed for all input neurons
        int inputCount = 0;
        for(const auto& L : graph.layers){
            if(L.kind != LayerKind::Input) continue;
            for(int idx : L.nodeIdx){
                if(idx < 0 || idx >= (int)graph.nodes.size()) continue;
                auto& n = graph.nodes[idx];
                if(!n.isInput) continue;
                
                // Set random amplitude (0.1 to 1.0)
                n.amp = 0.1f + (rand() / (float)RAND_MAX) * 0.9f;
                // Set random speed (0.1 to 1.0 Hz)
                n.speed = 0.1f + (rand() / (float)RAND_MAX) * 0.9f;
                // Enable live feed
                n.live = true;
                // Reset phase for immediate effect
                n.phase = 0.0f;
                inputCount++;
            }
        }
        Logf("Random live feed applied to %d input neurons", inputCount);
        return;
    }

    // Close celebration banner on any key press by resetting the 'done' flag visual
#ifdef USE_IMGUI
    if(gTrain.done){
        // Toggle done off/on to hide yellow bar; log remains.
        gTrain.done = false;
    }
#endif
}

// orbit handled in callbacks
void Engine::processCamera(){}

// main
#include <locale.h>
int main(){
    // Use C locale for consistent parsing and formatting
    setlocale(LC_ALL, "C");
    AppConfig cfg;
    char main_title[128];
    std::snprintf(main_title, sizeof(main_title), "NeuralT %s (C++/OpenGL - Neural Network Editor MB)", kAppVersion);
    cfg.title = main_title;
    Engine e;
    if(!e.init(cfg)) return 1;

    while(!glfwWindowShouldClose((GLFWwindow*)e.window)){
        e.update();
        e.render();
    }
    e.shutdown();
    return 0;
}
