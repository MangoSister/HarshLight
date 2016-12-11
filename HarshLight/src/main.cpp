#include <string.h>
#include "Material.h"
#include "World.h"
#include "Camera.h"
#include "VoxelizeController.h"
#include "ModelRenderer.h"
#include "FrameBufferDisplay.h"
#include "Util.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

const char* APP_NAME = "HarshLight";
const uint32_t DEFAULT_WINDOW_WIDTH = 1280;
const uint32_t DEFAULT_WINDOW_HEIGHT = 720;
const uint32_t GL_VER_MAJOR = 4;
const uint32_t GL_VER_MINOR = 5;

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
	uint32_t fill = 0;
	World::GetInst().m_DefaultBlackTex = new Texture2d(16, 16, fill);
	World::GetInst().RegisterTexture2d("DEFAULT_BLACK", World::GetInst().m_DefaultBlackTex);
	fill = 0xFFFFFFFF;
	World::GetInst().m_DefaultWhiteTex = new Texture2d(16, 16, fill);
	World::GetInst().RegisterTexture2d("DEFAULT_WHITE", World::GetInst().m_DefaultWhiteTex);
	fill = 0xFF808080;
	World::GetInst().m_DefaultGrayTex = new Texture2d(16, 16, fill);
	World::GetInst().RegisterTexture2d("DEFAULT_GRAY", World::GetInst().m_DefaultGrayTex);
	fill = 0x00FF8080;
	World::GetInst().m_DefaultNormalTex = new Texture2d(16, 16, fill);
	World::GetInst().RegisterTexture2d("DEFAULT_NORMAL", World::GetInst().m_DefaultNormalTex);
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

	World::GetInst().Destroy();
    glfwTerminate();
    return 0;
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

    ShaderProgram* local_illum_shader = new ShaderProgram();
    local_illum_shader->AddVertShader("src/shaders/local_illum_vert.glsl");
    local_illum_shader->AddFragShader("src/shaders/local_illum_frag.glsl");
    local_illum_shader->LinkProgram();
    World::GetInst().RegisterShader(local_illum_shader);

    ShaderProgram* depth_only_shader = new ShaderProgram();
    depth_only_shader->AddVertShader("src/shaders/depth_vert.glsl");
    depth_only_shader->AddFragShader("src/shaders/depth_frag.glsl");
    depth_only_shader->LinkProgram();
    World::GetInst().RegisterShader(depth_only_shader);

    ShaderProgram* depth_display_shader = new ShaderProgram();
    depth_display_shader->AddVertShader("src/shaders/depth_display_vert.glsl");
    depth_display_shader->AddFragShader("src/shaders/depth_display_frag.glsl");
    depth_display_shader->LinkProgram();
    World::GetInst().RegisterShader(depth_display_shader);

    ShaderProgram* depth_cube_shader = new ShaderProgram();
    depth_cube_shader->AddVertShader("src/shaders/depth_cube_vert.glsl");
    depth_cube_shader->AddGeomShader("src/shaders/depth_cube_geom.glsl");
    depth_cube_shader->AddFragShader("src/shaders/depth_cube_frag.glsl");
    depth_cube_shader->LinkProgram();
    World::GetInst().RegisterShader(depth_cube_shader);

    ShaderProgram* depth_display_cube_shader = new ShaderProgram();
    depth_display_cube_shader->AddVertShader("src/shaders/display_cubemap_vert.glsl");
    depth_display_cube_shader->AddFragShader("src/shaders/display_cubemap_frag.glsl");
    depth_display_cube_shader->LinkProgram();
    World::GetInst().RegisterShader(depth_display_cube_shader);

    ShaderProgram* framebuffer_display_shader = new ShaderProgram();
    framebuffer_display_shader->AddVertShader("src/shaders/framebuffer_color_vert.glsl");
    framebuffer_display_shader->AddFragShader("src/shaders/framebuffer_color_frag.glsl");
    framebuffer_display_shader->LinkProgram();
    World::GetInst().RegisterShader(framebuffer_display_shader);

	ShaderProgram* vct_shader = new ShaderProgram();
	vct_shader->AddVertShader("src/shaders/vct_grid_vert.glsl");
	vct_shader->AddFragShader("src/shaders/vct_grid_frag.glsl");
	vct_shader->LinkProgram();
	World::GetInst().RegisterShader(vct_shader);

    ShaderProgram* ds_geometry_shader = new ShaderProgram();
    ds_geometry_shader->AddVertShader("src/shaders/ds_geometry_pass_vert.glsl");
    ds_geometry_shader->AddFragShader("src/shaders/ds_geometry_pass_frag.glsl");
    ds_geometry_shader->LinkProgram();
    World::GetInst().RegisterShader(ds_geometry_shader);

    printf("Shaders compiling ended\n");

    printf("Loading scene started\n");

    /* --------------  Cameras  ----------- */
    const uint32_t voxel_dim = 256;
	const uint32_t light_injection_res = 1024;
    const glm::vec3 voxelize_center(0.0f, 300.0f, 0.0f);
    const glm::vec3 voxelize_extent(1000.0f, 450.0f, 600.0f);
    const float max_extent = std::max(voxelize_extent.x, std::max(voxelize_extent.y, voxelize_extent.z));
	const float voxel_scale = max_extent * 2.0f / static_cast<float>(voxel_dim);

    const float aspect = (float)DEFAULT_WINDOW_WIDTH / (float)DEFAULT_WINDOW_HEIGHT;
    {
        Actor* voxelize_cam_actor = new Actor();
		
        const float left = -1.0f * max_extent;
        const float right = 1.0f * max_extent;
        const float bottom = -1.0f * max_extent;
        const float top = 1.0f * max_extent;
        const float near = 0.0f;
        const float far = 2.0f * max_extent;
        Camera* cam = new Camera(left, right, bottom, top, near, far);
        cam->MoveTo(glm::vec3(0.0, max_extent, 0.0));
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
    
    Material* mat_depth_dir = new Material();
    mat_depth_dir->SetShader(depth_only_shader);
    World::GetInst().RegisterMaterial(mat_depth_dir);
    
    Material* mat_depth_cube = new Material();
    mat_depth_cube->SetShader(depth_cube_shader);
    World::GetInst().RegisterMaterial(mat_depth_cube);

    Actor* sceneActor = new Actor(); 

	//voxelization controller
	VoxelizeController* voxel_ctrl = new VoxelizeController(voxel_dim, light_injection_res, voxelize_center, voxelize_extent, World::GetInst().GetVoxelCamera());
	World::GetInst().m_VoxelizeController = voxel_ctrl;
	sceneActor->AddComponent(voxel_ctrl);

    ModelRenderer* sceneRenderer = new ModelRenderer(sceneModel);
    sceneActor->AddRenderer(sceneRenderer);
    sceneRenderer->MoveTo({ 0.0f, 0.0f, 0.0f });
    sceneRenderer->ScaleTo({ 0.5f, 0.5f, 0.5f });
    sceneRenderer->SetRenderPass((RenderPassFlag)(RenderPass::kVoxelize | RenderPass::kDirLightInjection | RenderPass::kPointLightInjection | RenderPass::kGeometry));
    for (Material*& mat_voxelize : sceneMaterials)
    {
        {
			mat_voxelize->AddTexture(voxel_ctrl->GetVoxelizeTex(VoxelChannel::TexVoxelAlbedo), VoxelizeController::s_VoxelChannelNames[VoxelChannel::TexVoxelAlbedo], TexUsage::kImageReadWrite, BINDING_POINT_START_VOXEL_IMG + VoxelChannel::TexVoxelAlbedo, 0);
			mat_voxelize->AddTexture(voxel_ctrl->GetVoxelizeTex(VoxelChannel::TexVoxelNormal), VoxelizeController::s_VoxelChannelNames[VoxelChannel::TexVoxelNormal], TexUsage::kImageReadWrite, BINDING_POINT_START_VOXEL_IMG + VoxelChannel::TexVoxelNormal, 0);
            mat_voxelize->SetShader(voxelize_shader);

            sceneRenderer->AddMaterial(RenderPass::kVoxelize, mat_voxelize);
        }

		{
			sceneRenderer->AddMaterial(RenderPass::kDirLightInjection, mat_depth_dir);
		}

        {
            sceneRenderer->AddMaterial(RenderPass::kPointLightInjection, mat_depth_cube);
        }

        {
            Material* mat_voxel_visual = new Material(*mat_voxelize);
           
			//mat_voxel_visual->DeleteAllTextures();	
			//mat_voxel_visual->AddTexture(voxel_ctrl->GetVoxelizeTex(VoxelChannel::TexVoxelAlbedo), VoxelizeController::s_VoxelChannelNames[VoxelChannel::TexVoxelAlbedo], TexUsage::kImageReadOnly, BINDING_POINT_START_VOXEL_IMG + VoxelChannel::TexVoxelAlbedo, 0);
			//mat_voxel_visual->AddTexture(voxel_ctrl->GetVoxelizeTex(VoxelChannel::TexVoxelNormal), VoxelizeController::s_VoxelChannelNames[VoxelChannel::TexVoxelNormal], TexUsage::kImageReadOnly, BINDING_POINT_START_VOXEL_IMG + VoxelChannel::TexVoxelNormal, 0);
			//mat_voxel_visual->AddTexture(voxel_ctrl->GetVoxelizeTex(VoxelChannel::TexVoxelRadiance), VoxelizeController::s_VoxelChannelNames[VoxelChannel::TexVoxelRadiance], TexUsage::kRegularTexture, BINDING_POINT_START_VOXEL_IMG + VoxelChannel::TexVoxelRadiance, 0);
			//mat_voxel_visual->AddTexture(voxel_ctrl->GetAnisoRadianceMipmap(0), "TexRadianceMipmap", TexUsage::kImageReadOnly, 4, 0);
			//mat_voxel_visual->SetShader(voxel_visualize_shader);
            

			//mat_voxel_visual->DeleteTexture(VoxelizeController::s_VoxelChannelNames[VoxelChannel::TexVoxelAlbedo]);
			//mat_voxel_visual->DeleteTexture(VoxelizeController::s_VoxelChannelNames[VoxelChannel::TexVoxelNormal]);
			//mat_voxel_visual->SetShader(local_illum_shader);
   //         mat_voxel_visual->SetFloatParam("Shininess", 10.0f);
			//char name[30];
			//for (uint32_t i = 0; i < LightManager::s_DirLightMaxNum; i++)
			//{
			//	memset(name, 0, 30);
			//	sprintf(name, "TexDirShadow[%u]", i);
			//	mat_voxel_visual->AddTexture2dDirect(voxel_ctrl->GetDirectionalDepthMap(i), name);
			//}

			//mat_voxel_visual->SetShader(vct_shader);
			//mat_voxel_visual->SetFloatParam("VoxelDim", static_cast<float>(voxel_dim));
			//mat_voxel_visual->SetFloatParam("VoxelScale", voxel_scale);
			//mat_voxel_visual->DeleteTexture(VoxelizeController::s_VoxelChannelNames[VoxelChannel::TexVoxelAlbedo]);
			//mat_voxel_visual->DeleteTexture(VoxelizeController::s_VoxelChannelNames[VoxelChannel::TexVoxelNormal]);
			//mat_voxel_visual->AddTexture(voxel_ctrl->GetVoxelizeTex(VoxelChannel::TexVoxelRadiance), "ImgRadianceLeaf", TexUsage::kRegularTexture, 0, 0);
			//mat_voxel_visual->SetFloatParam("Shininess", 10.0f);
			//char sampler_name[30];	
			//for (uint32_t i = 0; i < 6; i++)
			//{
			//	memset(sampler_name, 0, 30);
			//	sprintf(sampler_name, "ImgRadianceInterior[%d]", i);
			//	mat_voxel_visual->AddTexture(voxel_ctrl->GetAnisoRadianceMipmap(i), sampler_name, TexUsage::kRegularTexture, 0, 0);
			//}
			//for (uint32_t i = 0; i < LightManager::s_DirLightMaxNum; i++)
			//{
			//	memset(sampler_name, 0, 30);
			//	sprintf(sampler_name, "TexDirShadow[%u]", i);
			//	mat_voxel_visual->AddTexture2dDirect(voxel_ctrl->GetDirectionalDepthMap(i), sampler_name);
			//}

            mat_voxel_visual->SetShader(ds_geometry_shader);
            mat_voxel_visual->SetFloatParam("Shininess", 10.0f);
            mat_voxel_visual->DeleteTexture(VoxelizeController::s_VoxelChannelNames[VoxelChannel::TexVoxelAlbedo]);
            mat_voxel_visual->DeleteTexture(VoxelizeController::s_VoxelChannelNames[VoxelChannel::TexVoxelNormal]);

            sceneRenderer->AddMaterial(RenderPass::kGeometry, mat_voxel_visual);
			World::GetInst().RegisterMaterial(mat_voxel_visual);

        }
    }

    Model* quad = new Model(Model::Primitive::kQuad);
    //ModelRenderer* dir_light_depth_display = new ModelRenderer(quad);
    //dir_light_depth_display->SetRenderPass(RenderPass::kPost);
    //dir_light_depth_display->MoveTo({ -0.7f, 0.5f, 0.0f });
    //dir_light_depth_display->ScaleTo({ 1 / aspect, 1.0f, 1.0f });
    //Material* mat_depth_display = new Material();
    //mat_depth_display->SetShader(depth_display_shader);
    //mat_depth_display->AddTexture2dDirect(World::GetInst().m_VoxelizeController->GetDirectionalDepthMap(0), "TexDepth");
    //dir_light_depth_display->AddMaterial(RenderPass::kPost, mat_depth_display);
    //sceneActor->AddRenderer(dir_light_depth_display);

    //ModelRenderer* point_light_depth_display = new ModelRenderer(quad);
    //point_light_depth_display->SetRenderPass(RenderPass::kPost);
    //point_light_depth_display->MoveTo({ 0.7f, -0.5f, 0.0f });
    //point_light_depth_display->ScaleTo({ 1 / aspect, 1.0f, 1.0f });
    //Material* mat_depth_cube_display = new Material();
    //mat_depth_cube_display->SetShader(depth_display_cube_shader);
    //mat_depth_cube_display->AddTextureCubeDirect(World::GetInst().m_VoxelizeController->GetCubeDepthMap(0), "TexCube");
    //mat_depth_cube_display->SetI32Param("Face", 5);
    //point_light_depth_display->AddMaterial(RenderPass::kPost, mat_depth_cube_display);
    //sceneActor->AddRenderer(point_light_depth_display);

    World::GetInst().RegisterActor(sceneActor);

    /* --------------  Lights  ----------- */
    LightManager& light_manager = World::GetInst().GetLightManager();
    light_manager.SetAmbient(glm::vec3(0.15f, 0.15f, 0.15f)); 
    //light_manager.AddDirLight(DirLight(glm::vec3(0.424f, -0.8f, 0.424f), glm::vec4(0.8f, 0.77f, 0.55f, 1.2f)));
    light_manager.AddDirLight(DirLight(glm::vec3(0.2f, -0.9f, 0.1f), glm::vec4(0.8f, 0.77f, 0.55f, 1.5f)));
    light_manager.AddPointLight(PointLight(glm::vec3(0.0f, 50.0f, -200.0f), glm::vec4(0.0f, 1.0f, 1.0f, 10.0f)));
    light_manager.SetPointLightAtten(glm::vec3(1.0f, 0.01f, 0.01f));

    /* --------------  Frame Buffer Display  ----------- */
    // frame buffer display quad
   
    World::GetInst().RegisterModel(quad); 

    Actor* frameDisplayActor = new Actor();
  
    FrameBufferDisplay* voxelViewDisplay = new FrameBufferDisplay(quad, voxel_dim, true, true, GL_FILL);
    frameDisplayActor->AddRenderer(voxelViewDisplay);
    voxelViewDisplay->MoveTo({ 0.7f, 0.5f, 0.0f });
    voxelViewDisplay->ScaleTo({ 1 / aspect, 1.0f, 1.0f });
    voxelViewDisplay->SetRenderPass(RenderPass::kPost);
    Material* quad_mat = new Material();
    quad_mat->SetShader(framebuffer_display_shader);
    quad_mat->AddTexture2dDirect(voxelViewDisplay->GetColorBuffer(), "TexScreen");
    voxelViewDisplay->AddMaterial(RenderPass::kPost, quad_mat);

    World::GetInst().RegisterActor(frameDisplayActor);

    printf("Loading scene ended\n");
}