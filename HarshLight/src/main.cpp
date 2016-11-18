#include <string.h>
#include "glm/glm.hpp"
#include "Material.h"
#include "World.h"
#include "Camera.h"
#include "ModelRenderer.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

const char* APP_NAME = "HarshLight";
const uint32_t DEFAULT_WIDTH = 1920;
const uint32_t DEFAULT_HEIGHT = 1080;

void InitWorld(const char* scene_path);

int main(int argc, const char* argv[])
{
	const char* scene_path = nullptr;
	for (int32_t i = argc - 2; i >= 0; i -= 2)
	{
		if (strcmp(argv[i], "-i") == 0)
			scene_path = argv[i + 1];
	}

	if (!scene_path)
	{
		printf("usage:\n");
		printf("-i <scene file name>\n");
		exit(0);
	}

    GLFWwindow* window;

    /* Initialize the library */
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		exit(1);
	}

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

    World::GetInst().SetWindow(window);
	InitWorld(scene_path);


    World::GetInst().Start();

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
		World::GetInst().Update();
		
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
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
    sceneActor->AddComponent(new ModelRenderer(sceneModel, material));
    World::GetInst().AddActor(sceneActor);

	Actor* camActor = new Actor();	
	const float fovY = glm::radians(90.0f);
	const float aspect = 1.78f; // 16 : 9
	const float near = 0.01f;
	const float far = 10000.0f;
	camActor->AddComponent(new Camera(fovY, aspect, near, far));

	World::GetInst().AddActor(camActor);
}