#include <string.h>
#include "glm/glm.hpp"
#include "Material.h"
#include "World.h"
#include "Camera.h"
#include "ModelRenderer.h"
#include "Util.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

const char* APP_NAME = "HarshLight";
const uint32_t DEFAULT_WIDTH = 1920;
const uint32_t DEFAULT_HEIGHT = 1080;
const uint32_t GL_VER_MAJOR = 4;
const uint32_t GL_VER_MINOR = 5;

void InitWorld(const char* scene_path);

int main(int argc, const char* argv[])
{
	const char* scene_path = nullptr;
	uint8_t debug_mode = 0;
	for (int32_t i = argc - 2; i >= 0; i -= 2)
	{
		if (strcmp(argv[i], "-i") == 0)
			scene_path = argv[i + 1];
		else if (strcmp(argv[i], "-g") == 0)
			debug_mode = 1;
	}

	if (!scene_path)
	{
		printf("usage:\n");
		printf("-i <scene file name>\n");
		printf("-g <debug mode on/off> \n");
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
    window = glfwCreateWindow(DEFAULT_WIDTH, DEFAULT_HEIGHT, APP_NAME, NULL, NULL);
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

	printf("OpenGL version: %s\n", glGetString(GL_VERSION));
	printf("GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("Vendor: %s\n", glGetString(GL_VENDOR));
	printf("Renderer: %s\n", glGetString(GL_RENDERER));

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glDisable(GL_CULL_FACE);

	World::GetInst().SetWindow(window, DEFAULT_WIDTH, DEFAULT_HEIGHT);

	InitWorld(scene_path);

    World::GetInst().Start();

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Poll for and process events */
        glfwPollEvents();

		/* Render here */
		glClearColor(0.2f, 0.3f, 0.5f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		World::GetInst().Update();

		/* Swap front and back buffers */
		glfwSwapBuffers(window);

#ifdef _DEBUG
		if (debug_mode)
			assert(glGetError() == GL_NO_ERROR);
#endif
    }

    glfwTerminate();
    return 0;
}

void InitWorld(const char* scene_path)
{
    Material* material = new Material();
    material->AddVertShader("src/shaders/testvert.glsl");
    material->AddFragShader("src/shaders/testfrag.glsl");
    material->LinkProgram();
    World::GetInst().AddMaterial(material);

	Model* sceneModel = new Model(scene_path);
	World::GetInst().AddModel(sceneModel);

    Actor* sceneActor = new Actor();
	ModelRenderer* sceneRenderer = new ModelRenderer(sceneModel, material);
	sceneRenderer->MoveTo({ 0.0f, 0.0f, 0.0f });
	sceneRenderer->ScaleTo({ 0.5f, 0.5f, 0.5f });
    sceneActor->AddComponent(sceneRenderer);
    World::GetInst().AddActor(sceneActor);

	Model* quad = new Model(Model::Primitive::kTriangle);
	World::GetInst().AddModel(quad);

	Actor* quadActor = new Actor();
	ModelRenderer* quadRenderer = new ModelRenderer(quad, material);
	quadRenderer->MoveTo({ 0.0f, 0.0f, 2.0f });
	quadActor->AddComponent(quadRenderer);
	World::GetInst().AddActor(quadActor);

	Actor* camActor = new Actor();	
	const float fovY = glm::radians(90.0f);
	const float aspect = 1.78f; // 16 : 9
	const float near = 0.1f;
	const float far = 1000.0f;
	Camera* cam = new Camera(fovY, aspect, near, far);
	cam->MoveTo(glm::vec3(1.0f, 2.0f, 3.0f));
	cam->LookAt(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	cam->MoveTo(glm::vec3(0.0f, 1.0f, 0.0f));
	camActor->AddComponent(cam);

	World::GetInst().AddActor(camActor);
	World::GetInst().SetMainCamera(cam);
}