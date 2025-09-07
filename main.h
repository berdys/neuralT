#pragma once
// -*- coding: utf-8 -*-
// Main header for NeuralT + Neural Editor
// Type declarations used in main.cpp + paths and font helper.

#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <cstdio>
#include <unordered_map>

// Training state (moved from engine_patch.h)
struct TrainState{
    bool active=false;
    bool done=false;
    int totalEpochs=5000;
    int epoch=0;
    float lr=0.003f;
    float l1=0.0003f;
    float l2=0.0005f;
    float targetMSE=0.0f; // <=0 means ignore
    float lastMSE=0.0f;
    std::vector<float> mseHistory;
    double avgEpochMs=0.0;
    double startTime=0.0;
};
extern TrainState gTrain;
extern bool gTrainInterrupted;

// ─────────────────────────────────────────
// Paths and version
// ─────────────────────────────────────────
inline constexpr const char* kAppVersion             = "4.0.4";
inline constexpr const char* kEngineConfFile         = "engine.conf";
inline constexpr const char* kLessonsFile            = "lessons.json";
inline constexpr const char* kNetworkFile            = "network.json";
inline constexpr const char* kNetworkTrainedFile     = "network_trained.json";

// ─────────────────────────────────────────
// Minimal math
// ─────────────────────────────────────────
struct Vec2{ float x=0, y=0; };
struct Vec3{ float x=0, y=0, z=0; };

struct Mat4{
    float m[16] = {0};
    static Mat4 identity();
    static Mat4 perspective(float fovy, float aspect, float znear, float zfar);
    static Mat4 ortho(float l, float ri, float b, float t, float n, float f);
    static Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up);
    static Mat4 translate(const Vec3& v);
    static Mat4 rotateY(float a);
    static Mat4 rotateX(float a);
    static Mat4 scale(const Vec3& v);
    Mat4 operator*(const Mat4& o) const;
};

struct Timer{
    double now=0, last=0, dt=0;
    void update(double currentTime);
};

struct Bounds3{
    Vec3 min{ 1e9f, 1e9f, 1e9f };
    Vec3 max{-1e9f,-1e9f,-1e9f };
    void expand(const Vec3& p);
};

// ─────────────────────────────────────────
// Shaders / meshes
// ─────────────────────────────────────────
struct Shader{
    unsigned int program = 0;
    bool compile(const char* vs, const char* fs, const char* gs, std::string* logOut);
    void use() const;
    void destroy();
    int  uniform(const char* name) const; // GLint
};

struct Vertex{ Vec3 pos; Vec3 normal; Vec2 uv; };

struct Mesh{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    unsigned int vao=0, vbo=0, ebo=0;
    Bounds3 bounds;
    void upload();
    void draw() const;
    void destroy();
};

namespace Procedural{
    Mesh makeCylinder(float R, float H, int seg, bool capped);
    Mesh makeBlade(float rootW, float tipW, float L, float T, int slices);
    Mesh makeGear(float innerR, float outerR, float toothDepth, int teeth, float width);
    Mesh merge(const std::vector<Mesh>& meshes);
    Mesh makeLineList(const std::vector<Vec3>& pts, const std::vector<uint32_t>& inds);
    // Simple 3D arrow prism aligned along +Z (length axis)
    Mesh makeArrowPrism(float length, float width, float thickness, float headLen);
    // Streamlined wedge prism (triangular ship)
    Mesh makeWedgePrism(float length, float width, float thickness);
    // Axis-aligned box centered at origin
    Mesh makeBox(float sx, float sy, float sz);
}

// ─────────────────────────────────────────
// Camera, UI theme
// ─────────────────────────────────────────
struct OrbitCamera{
    float yaw=0, pitch=0, distance=3;
    Vec3  target{0,0,0};
    Mat4 view() const;
    void orbit(float dx, float dy);
    void dolly(float dd);
    void pan(float dx, float dy);
};

struct UITheme{
    std::string name;
    Vec3 bg, panel, accent, grid, text, mesh;
    // Neural editor colors
    Vec3 nInput, nHidden, nOutput, nRing, nHighlight;
    // Edge gradient (start -> end) and style
    Vec3 edgeStart, edgeEnd;
    float edgeAlpha=0.85f;
    float edgeBase=0.6f, edgeScale=0.8f;

    // Five cohesive themes
    static UITheme NeonNoir();
    static UITheme DeepSpace();
    static UITheme SolarFlare();
    static UITheme Sunset();
    static UITheme EmberForge();
    static UITheme Crimson();
    static UITheme Ruby();
    static UITheme Cyberpunk();
    static UITheme Aurora();
    static UITheme ForestMist();
    static UITheme OceanWave();
    static UITheme MonoSlate();
};

// ─────────────────────────────────────────
// Turbine model
// ─────────────────────────────────────────
struct TurbineModel{
    struct Params{
        float rpm=60;
        float current_rpm=0.0f;
        int   blades=3;
        float shaftLength=1.6f, shaftRadius=0.08f;
        float hubRadius=0.2f;
        float bladeLength=0.9f;
        float bladeRootWidth=0.25f, bladeTipWidth=0.12f, bladeThickness=0.02f;
    } params;
    Mesh hub, shaft, blades, gear, rivets, rotor, outerGear, outerGearR, pipes, schematicLines;
    int rotorTeeth=24, outerTeeth=36, outerTeethR=36;
    float outerGearOffset=0.0f, outerGearOffsetR=0.0f;
    void rebuild();
    void destroy();
};

// ─────────────────────────────────────────
// Sieć neuronowa – dane i graf
// ─────────────────────────────────────────
enum class LayerKind { Input, Hidden, Output };
enum class Act { Linear, ReLU, Sigmoid, Tanh };
namespace Activation{
    static constexpr Act Linear  = Act::Linear;
    static constexpr Act ReLU    = Act::ReLU;
    static constexpr Act Sigmoid = Act::Sigmoid;
    static constexpr Act Tanh    = Act::Tanh;
}
const char* ActName(Act a);
Act NextAct(Act a);

struct Layer{
    LayerKind kind = LayerKind::Hidden;
    std::string name;
    Act act = Act::Linear;
    std::vector<int> nodeIdx;
    float x0=0,y0=0,x1=0,y1=0;
};

struct Neuron{
    int id=0;
    bool isInput=false;
    std::string label;
    std::string desc = "default description";
    float value=0, speed=0, bias=0, radius=14;
    float amp=1.0f; // amplitude for live feed (0..1)
    bool live=false; // per-input live feed toggle
    float x=0, y=0;
    float vx=0, vy=0;
    float phase=0;
};

struct Edge{ int a=0, b=0; float w=0; };

struct NeuralGraph{
    std::vector<Neuron> nodes;
    std::vector<Edge> edges;
    std::vector<Layer> layers;

    int selected=-1;
    int currentLayer=0;
    bool liveInputs=false;
    char editBuf[256]{};
    float lastCanvasW=0, lastCanvasH=0;

    Neuron* findAt(float mx, float my);
    void buildDemo();
    void layoutLayers(float W, float H);
    void tickLiveInputs(float dt);
    void stepLayout(float dt, float W, float H);
    void drawCanvas(bool& open, float width, float height);
    // Lightweight persistent cache for topology lookups (id -> indices and edge lists)
    struct GraphCache{
        std::unordered_map<int,int> id2idx;
        std::unordered_map<int,std::vector<int>> outEdges; // id -> edge indices where id is src
        std::unordered_map<int,std::vector<int>> inEdges;  // id -> edge indices where id is dst
        uint64_t version = 0; // matches topoVersion when valid
        bool     valid   = false;
    } cache;
    uint64_t topoVersion = 0; // bump on any nodes/edges topology change
};

// ─────────────────────────────────────────
// Application
// ─────────────────────────────────────────
struct UITheme; // forward above
struct Renderer{
    UITheme theme;
    Shader meshShader, lineShader;
    unsigned int gridVao=0, gridVbo=0;
    bool init(const UITheme& t);
    void drawGrid(const Mat4& vp) const;
    void drawMesh(const Mesh& m, const Mat4& mvp, const Mat4& model, const Vec3& color, float metallic) const;
    void drawLines(const Mesh& lineList, const Mat4& mvp, const Vec3& color, float thickness) const;
    void drawMeshFlat(const Mesh& m, const Mat4& mvp, const Vec3& color) const;
    void destroy();
};

struct InputState{
    bool mouseDownL=false, mouseDownM=false, mouseDownR=false;
    bool shift=false, alt=false, ctrl=false;
    double mouseX=0, mouseY=0;
};

struct AppConfig{
    std::string title = "NeuralX";
    int width = 1280, height = 800;
    bool vsync = true;
    bool rotate = false;
    bool showSchematic = false;
    bool showUI = true;
};

struct Engine{
    void* window = nullptr;
    Renderer renderer;
    TurbineModel turbine;
    OrbitCamera cam;
    Timer timer;
    AppConfig cfg;
    NeuralGraph graph;
    float rotorAngle = 0;
    bool networkDirty = false; // true when topology/weights/biases changed since last save
    bool topDown2D = false;    // F8: toggle top-down orthographic view
    Vec3 topDownCenter{0.0f, 0.0f, 0.0f}; // pan/center for Simulator Map
    bool cameraFollow = false; // F8: follow selected player
    Vec3 topDownCenterTarget{0.0f, 0.0f, 0.0f};
    // Input mapping selection per input neuron (index into parameter sources)
    std::vector<int> inputParamSelection;

    // Player entity and AI controller types (F8 mode)
struct PlayerEntity{
        std::string name;
        Vec3  pos{0,0,-4};
        float yaw = 0.0f;     // radians, 0 looks along +Z
        Vec3  color{0.95f,0.85f,0.2f};
        int   aiIndex = -1;   // -1 => manual
    };
    struct Bush{ Vec3 pos{0,0,0}; Vec2 size{0.6f,0.6f}; };

    enum class ShapeType { Wedge, Box, Cylinder };
    struct ShapeEntity{
        std::string name;
        ShapeType type = ShapeType::Wedge;
        Vec3  color{0.4f,0.8f,1.0f};
        Vec3  pos{0,0,0};
        float yaw = 0.0f;
        Vec3  scale{1.0f,1.0f,1.0f};
        int   inputIdDist = -1; // bound network input id for Dist<name>
    };
    struct AIController{
        std::string name;
        std::string path;
        // Dedicated model independent from editor graph
        NeuralGraph net;
        // Mapping heuristic: indices of first 3 input neurons (or by labels)
        std::vector<int> inputAbsIdx; // absolute node indices of inputs in net.nodes
        std::vector<int> outputAbsIdx; // absolute node indices for outputs
        bool loaded=false;
    };
    // World meshes
    Mesh playerMesh;   // reused for all players (wedge)
    Mesh bushMesh;     // unit box (scaled in shader via model matrix)
    Mesh cylinderMesh; // unit cylinder
    // World entities
    std::vector<PlayerEntity> players;
    std::vector<Bush> bushes;
    std::vector<ShapeEntity> shapes;
    // AI controllers available (loaded from saved/)
    std::vector<AIController> aiControllers;
    int selectedPlayer = -1;
    int selectedShape = -1;
    // Legacy single-player mapping to editor graph (manual inputs)
    int playerInputIdX = -1, playerInputIdY = -1, playerInputIdZ = -1, playerInputIdYaw = -1;

    void loadTurbinConfDefaults(const std::string& path);
    bool init(const AppConfig& cfgIn);
    void update();
    Mat4 proj() const;
    Mat4 view() const;
    void render();
    void shutdown();
    void onResize(int w, int h);
    void onMouseButton(int button, int action, int mods);
    void onCursorPos(double x, double y);
    void onScroll(double dx, double dy);
    void onKey(int key, int scancode, int action, int mods);
    void processCamera();
    // Helpers for F8 mode
    void updateTopDownPlayers(float dt);
    void ensurePlayerInputs();
    void ensurePlayerYawInput();
    void pushPlayerCoordsToInputs();
    void updateShapeDistanceInputs();
    void applyInputMappings();
};

#ifdef USE_IMGUI
// Dear ImGui – load fonts with extended Latin range
#include "imgui.h"
inline void SetupDefaultFonts(float size_px = 16.0f){
    ImGuiIO& io = ImGui::GetIO();
    ImFontConfig cfg; cfg.OversampleH = 3; cfg.OversampleV = 2; cfg.PixelSnapH = false;
    // Include Basic Latin, Latin-1 Supplement and Latin Extended-A (covers Polish)
    static const ImWchar ranges[] = { 0x0020, 0x00FF, 0x0100, 0x017F, 0 };
    const char* candidates[] = {
        "DejaVuSans.ttf", "fonts/DejaVuSans.ttf", "NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        // Windows common fonts
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf"
    };
    bool loaded=false;
    for(const char* p : candidates){
        FILE* fp = std::fopen(p, "rb");
        if(fp){ std::fclose(fp); io.Fonts->AddFontFromFileTTF(p, size_px, &cfg, ranges); loaded = true; break; }
    }
    if(!loaded) io.Fonts->AddFontDefault();
}
#endif // USE_IMGUI
