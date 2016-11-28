#include <string.h>
#include "glm/glm.hpp"
#include "Material.h"
#include "World.h"
#include "Camera.h"
#include "VoxelizeController.h"
#include "ModelRenderer.h"
#include "FrameBufferDisplay.h"
#include "Util.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>

const char* APP_NAME = "HarshLight";
const uint32_t DEFAULT_WINDOW_WIDTH = 1920;
const uint32_t DEFAULT_WINDOW_HEIGHT = 1080;
const uint32_t GL_VER_MAJOR = 4;
const uint32_t GL_VER_MINOR = 5;

void CreateCRTestScene();
void CreateWorld(const char* scene_path, float mouse_sensitivity);

int main(int argc, char* argv[])
{
	char* scene_path = nullptr;
	uint8_t debug_mode = 0;
	float mouse_sensitivity = 0.01f;
	for (int32_t i = argc - 2; i >= 0; i -= 2)
	{
		if (strcmp(argv[i], "-i") == 0)
		{
			scene_path = argv[i + 1];
			size_t len = strlen(scene_path);
			for (size_t i = 0; i < len; i++)
			{
				if (scene_path[i] == '\\')
					scene_path[i] = '/';
			}
		}
		else if (strcmp(argv[i], "-g") == 0)
			debug_mode = 1;
		else if (strcmp(argv[i], "-m") == 0)
			mouse_sensitivity = static_cast<float>(atof(argv[i + 1]));
	}

	if (!scene_path)
	{
		printf("usage:\n");
		printf("-i <scene file name>\n");
		printf("-g <debug mode on/off> \n");
		printf("-m <mouse sensitivity> \n");
		exit(0);
	}

    GLFWwindow* window;

    /* Initialize the library */
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		exit(1);
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, GL_VER_MAJOR);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, GL_VER_MINOR);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	if (debug_mode)
		glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);


    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT, APP_NAME, NULL, NULL);
    if (!window)
    {
		fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
		exit(1);
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK)
	{
		fprintf(stderr, "Failed to initialize GLEW\n");
		glfwTerminate();
		exit(1);
	}

	if (debug_mode)
	{
		GLint succ;
		glGetIntegerv(GL_CONTEXT_FLAGS, &succ);
		if (succ & GL_CONTEXT_FLAG_DEBUG_BIT)
		{
			printf("debug Mode activated\n");
			glEnable(GL_DEBUG_OUTPUT);
			glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
			glDebugMessageCallback(glDebugOutput, nullptr);
			glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
		}
		else
			fprintf(stderr, "fail to activate debug mode\n");
	}
    printf("------- OpenGL Info -------\n");
	printf("OpenGL version: %s\n", glGetString(GL_VERSION));
	printf("GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("Vendor: %s\n", glGetString(GL_VENDOR));
	printf("Renderer: %s\n", glGetString(GL_RENDERER));
    printf("---------------------------\n");
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    World::GetInst().SetWindow(window, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
    //CreateCRTestScene();    
	CreateWorld(scene_path, mouse_sensitivity);

    World::GetInst().Start();

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
#ifdef _DEBUG
        if (debug_mode)
            assert(glGetError() == GL_NO_ERROR);
#endif

        /* Poll for and process events */
        glfwPollEvents();

		/* Render here */
		World::GetInst().MainLoop();

		/* Swap front and back buffers */
		glfwSwapBuffers(window);

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            break;
    }

    glfwTerminate();
    return 0;
}

void CreateCRTestScene()
{
    const uint32_t test_dim = 16;
    /* --------------  Shaders  ----------- */
    ShaderProgram* cr_shader = new ShaderProgram();
    cr_shader->AddVertShader("src/shaders/cr_vert.glsl");
    cr_shader->AddGeomShader("src/shaders/cr_geom.glsl");
    cr_shader->AddFragShader("src/shaders/cr_frag.glsl");
    cr_shader->LinkProgram();
    World::GetInst().RegisterShader(cr_shader);

    ShaderProgram* sr_shader = new ShaderProgram();
    sr_shader->AddVertShader("src/shaders/sr_vert.glsl");
    sr_shader->AddFragShader("src/shaders/sr_frag.glsl");
    sr_shader->LinkProgram();
    World::GetInst().RegisterShader(sr_shader);

    ShaderProgram* framebuffer_display_shader = new ShaderProgram();
    framebuffer_display_shader->AddVertShader("src/shaders/framebuffer_color_vert.glsl");
    framebuffer_display_shader->AddFragShader("src/shaders/framebuffer_color_frag.glsl");
    framebuffer_display_shader->LinkProgram();
    World::GetInst().RegisterShader(framebuffer_display_shader);

    printf("Shaders compiling success\n");

    /* --------------  Cameras  ----------- */
    const float aspect = (float)DEFAULT_WINDOW_WIDTH / (float)DEFAULT_WINDOW_HEIGHT;
    {
        Actor* cam_actor = new Actor();
        const float left = -1.0f;
        const float right = 1.0f;
        const float bottom = -1.0f;
        const float top = 1.0f;
        const float near = -1.0f;
        const float far = 1.0f;
        Camera* cam = new Camera(left, right, bottom, top, near, far);
        cam->MoveTo(glm::vec3(0.0f, 0.0f, 0.0f));
        cam->LookAtDir(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f));

        cam_actor->AddComponent(cam);
        World::GetInst().RegisterActor(cam_actor);
        World::GetInst().SetMainCamera(cam);
        World::GetInst().SetVoxelCamera(cam);
    }

    
    /* --------------  Test Shape  ----------- */
    Model* tri = new Model(Model::Primitive::kTriangle);
    World::GetInst().RegisterModel(tri);

    Actor* cr_triActor = new Actor();

    ModelRenderer* cr_renderer = new ModelRenderer(tri);
    //cr_renderer->ScaleTo(glm::vec3(1.0, 0.5, 1.0));
    cr_triActor->AddRenderer(cr_renderer);
    cr_renderer->SetRenderPass(RenderPass::kRegular);
    Material* cr_mat = new Material();
    cr_mat->SetShader(cr_shader);
    cr_renderer->AddMaterial(RenderPass::kRegular, cr_mat);


    /* --------------  Controller  ----------- */
    VoxelizeController* voxel_ctrl = new VoxelizeController(test_dim, 0.0f, World::GetInst().GetVoxelCamera());
    cr_triActor->AddComponent(voxel_ctrl);

    World::GetInst().RegisterActor(cr_triActor);

    glm::vec4 a(0.0, 0.5, 0, 1);
    a = cr_renderer->GetTransform() * a;
    a = World::GetInst().GetMainCamera()->GetViewMtx() * a;
    a = World::GetInst().GetMainCamera()->GetProjMtx() * a;

    Actor* sr_triActor = new Actor();

    ModelRenderer* sr_renderer = new ModelRenderer(tri);
    //sr_renderer->ScaleTo(glm::vec3(1.0, 0.5, 1.0));
    sr_triActor->AddRenderer(sr_renderer);
    sr_renderer->SetRenderPass(RenderPass::kRegular);
    Material* sr_mat = new Material();
    sr_mat->SetShader(sr_shader);
    sr_renderer->AddMaterial(RenderPass::kRegular, sr_mat);

    World::GetInst().RegisterActor(sr_triActor);


    /* --------------  Frame Buffer Display  ----------- */
    Model* quad = new Model(Model::Primitive::kQuad);
    World::GetInst().RegisterModel(quad);
    
    {
        Actor* rasterTriDisplay = new Actor();

        FrameBufferDisplay* framBufDisplay = new FrameBufferDisplay(quad, test_dim, false, false, GL_FILL);
        rasterTriDisplay->AddRenderer(framBufDisplay);
        framBufDisplay->MoveTo({ 0.7f, 0.5f, 0.0f });
        framBufDisplay->ScaleTo({ 1 / aspect, 1.0f, 1.0f });
        framBufDisplay->SetRenderPass(RenderPass::kPost);
        Material* frame_display_mat = new Material();
        frame_display_mat->SetShader(framebuffer_display_shader);
        frame_display_mat->AddTexture(framBufDisplay->GetColorBuffer(), "TexScreen");
        framBufDisplay->AddMaterial(RenderPass::kPost, frame_display_mat);

        World::GetInst().RegisterActor(rasterTriDisplay);
    }

    {
        const uint32_t ref_dim = 256;
        Actor* lineTriDisplay = new Actor();

        FrameBufferDisplay* framBufDisplay = new FrameBufferDisplay(quad, ref_dim, false, false, GL_LINE);
        lineTriDisplay->AddRenderer(framBufDisplay);
        framBufDisplay->MoveTo({ 0.7f, 0.5f, 0.0f });
        framBufDisplay->ScaleTo({ 1 / aspect, 1.0f, 1.0f });
        framBufDisplay->SetRenderPass(RenderPass::kPost);
        Material* frame_display_mat = new Material();
        frame_display_mat->SetShader(framebuffer_display_shader);
        frame_display_mat->AddTexture(framBufDisplay->GetColorBuffer(), "TexScreen");
        framBufDisplay->AddMaterial(RenderPass::kPost, frame_display_mat);

        World::GetInst().RegisterActor(lineTriDisplay);
    }
}

void CreateWorld(const char* scene_path, float mouse_sensitivity)
{
    /* --------------  Shaders  ----------- */

    ShaderProgram* voxelize_shader = new ShaderProgram();
    voxelize_shader->AddVertShader("src/shaders/voxelize_vert.glsl");
    voxelize_shader->AddGeomShader("src/shaders/voxelize_geom.glsl");
    voxelize_shader->AddFragShader("src/shaders/voxelize_frag.glsl");
    voxelize_shader->LinkProgram();
    World::GetInst().RegisterShader(voxelize_shader);



    ShaderProgram* voxel_visualize_shader = new ShaderProgram();
    voxel_visualize_shader->AddVertShader("src/shaders/voxel_visualize_vert.glsl");
    voxel_visualize_shader->AddFragShader("src/shaders/voxel_visualize_frag.glsl");
    voxel_visualize_shader->LinkProgram();
    World::GetInst().RegisterShader(voxel_visualize_shader);



    ShaderProgram* diffuse_shader = new ShaderProgram();
    diffuse_shader->AddVertShader("src/shaders/diffuse_vert.glsl");
    diffuse_shader->AddFragShader("src/shaders/diffuse_frag.glsl");
    diffuse_shader->LinkProgram();
    World::GetInst().RegisterShader(diffuse_shader);



    ShaderProgram* framebuffer_display_shader = new ShaderProgram();
    framebuffer_display_shader->AddVertShader("src/shaders/framebuffer_color_vert.glsl");
    framebuffer_display_shader->AddFragShader("src/shaders/framebuffer_color_frag.glsl");
    framebuffer_display_shader->LinkProgram();
    World::GetInst().RegisterShader(framebuffer_display_shader);

    printf("Shaders compiling success\n");

    /* --------------  Cameras  ----------- */
    const uint32_t voxelDim = 256;
    const float voxelize_extent = 1100.0f;
    const float aspect = (float)DEFAULT_WINDOW_WIDTH / (float)DEFAULT_WINDOW_HEIGHT;
    {
        Actor* voxelize_cam_actor = new Actor();
		
        const float left = -1.0f * voxelize_extent;
        const float right = 1.0f * voxelize_extent;
        const float bottom = -1.0f * voxelize_extent;
        const float top = 1.0f * voxelize_extent;
        const float near = 0.0f;
        const float far = 2.0f * voxelize_extent;
        Camera* cam = new Camera(left, right, bottom, top, near, far);
        cam->MoveTo(glm::vec3(0.0, voxelize_extent, 0.0));
        cam->LookAtDir(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

        glm::vec4 a = glm::vec4(1000, -4000, -1000, 1);
        auto view = cam->GetViewMtx();
        auto proj = cam->GetProjMtx();
        a = view * a;
        a = proj * a;

        voxelize_cam_actor->AddComponent(cam);
        World::GetInst().RegisterActor(voxelize_cam_actor);
        World::GetInst().SetVoxelCamera(cam);
    }

    {
        Actor* fps_cam_actor = new Actor();
        const float fovY = glm::radians(45.0f);
        const float near = 0.1f;
        const float far = 10000.0f;
        const float move_speed = 100.0f;
        printf("------- FPS Camera Info -------\n");
        printf("fovY: %f\n", fovY);
        printf("aspect ratio: %f\n", aspect);
        printf("near plane z: %f\n", near);
        printf("far plane z: %f\n", far);
        printf("free move speed: %f\n", move_speed);
        printf("fps mouse sensitivity: %f\n", mouse_sensitivity);
        printf("max pitch angle: 89 degree\n");
        printf("---------------------------\n");
        Camera* cam = new Camera(fovY, aspect, near, far);
        cam->MoveTo(glm::vec3(0.0f, 0.0f, 0.0f));
        cam->LookAtDir(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        cam->SetFreeMoveSpeed(move_speed);

        fps_cam_actor->AddComponent(cam);
        World::GetInst().RegisterActor(fps_cam_actor);
        World::GetInst().SetMainCamera(cam);
    }

    World::GetInst().SetMouseSensitivity(mouse_sensitivity);

    /* --------------  Scene  ----------- */
    //hardcode test world...:p
    Model* sceneModel = new Model(scene_path);
    World::GetInst().RegisterModel(sceneModel);
    std::vector<Material*> sceneMaterials = World::GetInst().LoadDefaultMaterialsForModel(sceneModel);
    Actor* sceneActor = new Actor(); 

    //voxel grid texture3d
    Texture3dCompute* voxelTex = new Texture3dCompute(voxelDim, voxelDim, voxelDim);
	World::GetInst().m_VoxelizeTex = voxelTex;
    World::GetInst().RegisterTexture3d(voxelTex);

	//voxelization controller
	VoxelizeController* voxel_ctrl = new VoxelizeController(voxelDim, voxelize_extent, World::GetInst().GetVoxelCamera());
	sceneActor->AddComponent(voxel_ctrl);

    ModelRenderer* sceneRenderer = new ModelRenderer(sceneModel);
    sceneActor->AddRenderer(sceneRenderer);
    sceneRenderer->MoveTo({ 0.0f, 0.0f, 0.0f });
    sceneRenderer->ScaleTo({ 0.5f, 0.5f, 0.5f });
    sceneRenderer->SetRenderPass((RenderPassFlag)(RenderPass::kVoxelize | RenderPass::kRegular));
    for (Material*& mat_voxelize : sceneMaterials)
    {
        {
            mat_voxelize->AddTexture(voxelTex, "TexVoxel", TexUsage::kImageWriteOnly, BINDING_POINT_VOXEL_IMG);
            mat_voxelize->SetShader(voxelize_shader);
            GLuint shader_obj = mat_voxelize->GetShader()->GetProgram();
            sceneRenderer->AddMaterial(RenderPass::kVoxelize, mat_voxelize);
        }

        {
            Material* mat_voxel_visual = new Material(*mat_voxelize);
            mat_voxel_visual->DeleteAllTextures();
            mat_voxel_visual->AddTexture(voxelTex, "TexVoxel", TexUsage::kRegularTexture, 0);
            mat_voxel_visual->SetShader(voxel_visualize_shader);

			//mat_voxel_visual->DeleteTexture("TexVoxel");
			//mat_voxel_visual->SetShader(diffuse_shader);

            
            GLuint shader_obj = mat_voxel_visual->GetShader()->GetProgram();
            //set voxel camera matrices
            glm::mat4x4 voxel_view = World::GetInst().GetVoxelCamera()->GetViewMtx();
            glm::mat4x4 voxel_proj = World::GetInst().GetVoxelCamera()->GetProjMtx();
           // glUniform4fv(glGetUniformLocation(shader_obj, "CamVoxelViewMtx"), 1, glm::value_ptr(voxel_view));
           // glUniform4fv(glGetUniformLocation(shader_obj, "CamVoxelProjMtx"), 1, glm::value_ptr(voxel_proj));
            sceneRenderer->AddMaterial(RenderPass::kRegular, mat_voxel_visual);
        }
    }
    World::GetInst().RegisterActor(sceneActor);

    /* --------------  Frame Buffer Display  ----------- */
    // frame buffer display quad
    Model* quad = new Model(Model::Primitive::kQuad);
    World::GetInst().RegisterModel(quad);

    Actor* frameDisplayActor = new Actor();
  
    FrameBufferDisplay* voxelViewDisplay = new FrameBufferDisplay(quad, voxelDim, true, true, GL_FILL);
    frameDisplayActor->AddRenderer(voxelViewDisplay);
    voxelViewDisplay->MoveTo({ 0.7f, 0.5f, 0.0f });
    voxelViewDisplay->ScaleTo({ 1 / aspect, 1.0f, 1.0f });
    voxelViewDisplay->SetRenderPass(RenderPass::kPost);
    Material* quad_mat = new Material();
    quad_mat->SetShader(framebuffer_display_shader);
    quad_mat->AddTexture(voxelViewDisplay->GetColorBuffer(), "TexScreen");
    voxelViewDisplay->AddMaterial(RenderPass::kPost, quad_mat);

    World::GetInst().RegisterActor(frameDisplayActor);
}