#include "World.h"
#include <cassert>

World::~World()
{
    for (Actor*& actor : m_Actors)
    {
        if (actor)
        {
            delete actor;
            actor = nullptr;
        }
    }

	for (Model*& model : m_Models)
	{
		if (model)
		{
			delete model;
			model = nullptr;
		}
	}

    for (Material*& material : m_Materials)
    {
        if (material)
        {
            delete material;
            material = nullptr;
        }
    }
}

void World::MouseCallback(GLFWwindow * window, double xpos, double ypos)
{
	World& world = World::GetInst();
	static double s_LastMouseX = 0.5 * static_cast<double>(world.m_ScreenWidth);
	static double s_LastMouseY = 0.5 * static_cast<double>(world.m_ScreenHeight);

	double xoffset = xpos - s_LastMouseX;
	double yoffset = s_LastMouseY - ypos; // Reversed since y-coordinates range from bottom to top
	s_LastMouseX = xpos;
	s_LastMouseY = ypos;

	const double sensitivity = 0.01;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	static double yaw = 0.0;
	static double pitch = 0.0;
	printf("%f %f\n", yaw, pitch);
	yaw += xoffset;
	yaw = fmod(yaw, 360);
	pitch += yoffset;
	if (pitch > 89)
		pitch = 89;
	if (pitch < -89)
		pitch = -89;

	Camera* main_camera = world.GetMainCamera();
	if (main_camera)
	{
		glm::vec3 forward;
		forward.z = std::cos(glm::radians(pitch)) * std::cos(glm::radians(yaw));
		forward.y = std::sin(glm::radians(pitch));
		forward.x = std::cos(glm::radians(pitch)) * std::sin(glm::radians(yaw));
		forward = normalize(forward);
		main_camera->LookAtDir(forward, glm::vec3(0.0f, 1.0f, 0.0f));
	}
}

void World::SetWindow(GLFWwindow* window, uint32_t width, uint32_t height)
{
    m_Window = window;
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(window, MouseCallback);
	m_ScreenWidth = width;
	m_ScreenHeight = height;
}

const ActorList& World::GetActors() const
{
    return m_Actors;
}

void World::AddActor(Actor* actor)
{
#ifdef _DEBUG
    assert(actor != nullptr);
#endif
    m_Actors.push_back(actor);
}

const ModelList& World::GetModels() const
{
	return m_Models;
}

void World::AddModel(Model* model)
{
#if _DEBUG
	assert(model != nullptr);
#endif
	m_Models.push_back(model);
}

const MaterialList& World::GetMaterials() const
{
    return m_Materials;
}

void World::AddMaterial(Material* material)
{
#ifdef _DEBUG
    assert(material != nullptr);
#endif
    m_Materials.push_back(material);
}

void World::SetMainCamera(Camera * camera)
{
#ifdef _DEBUG
	assert(camera != nullptr);
#endif
	m_MainCamera = camera;
}

Camera* World::GetMainCamera()
{
	return m_MainCamera;
}

void World::Start()
{
    for (Actor* actor : m_Actors)
    {
#ifdef _DEBUG
        assert(actor != nullptr);
#endif
        actor->Start();
    }

	m_LastTime = std::chrono::system_clock::now();
	m_CurrTime = std::chrono::system_clock::now();
}

void World::Update()
{
	m_CurrTime = std::chrono::system_clock::now();
	float elapsed = static_cast<float>((std::chrono::duration<double>(m_CurrTime - m_LastTime)).count());
	//fps: 1 / elapsed
	m_LastTime = m_CurrTime;

	if (m_MainCamera)
		m_MainCamera->UpdateCamMtx();
	else
		fprintf(stderr, "WARNING: MainCamera is null\n");

    for (Actor* actor : m_Actors)
    {
        assert(actor != nullptr);
        actor->Update(elapsed);
    }
}

int World::GetKey(int key)
{
    return glfwGetKey(m_Window, key);
}