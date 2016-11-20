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

    for (auto it = m_Texture2ds.begin(); it != m_Texture2ds.end(); it++)
    {
        //tex2d
        if (it->second)
        {
            delete (it->second);
            it->second = nullptr;
        }
    }
}

void World::MouseCallback(GLFWwindow * window, double xpos, double ypos)
{
	World& world = World::GetInst();
	static double s_LastMouseX = 0.5 * static_cast<double>(world.m_ScreenWidth);
	static double s_LastMouseY = 0.5 * static_cast<double>(world.m_ScreenHeight);

	double xoffset = xpos - s_LastMouseX;
	double yoffset = ypos - s_LastMouseY;
	s_LastMouseX = xpos;
	s_LastMouseY = ypos;

	const double SENSITIVITY = 0.01;
	xoffset *= SENSITIVITY;
	yoffset *= SENSITIVITY;

	static float yaw = 0.0f;
	static float pitch = 0.0f;
    const float MAX_PITCH = 89.0f;
	//printf("%f %f\n", yaw, pitch);
	yaw += static_cast<float>(xoffset);
	yaw = fmod(yaw, 360.0f);  
	pitch += static_cast<float>(yoffset);
    pitch = std::fmin(std::fmax(pitch, -MAX_PITCH), MAX_PITCH);

	Camera* main_camera = world.GetMainCamera();
	if (main_camera)
	{
		glm::vec3 forward;
		forward.z = std::cosf(glm::radians(pitch)) * std::cosf(glm::radians(yaw));
		forward.y = std::sinf(glm::radians(pitch));
		forward.x = std::cosf(glm::radians(pitch)) * std::sinf(glm::radians(yaw));
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

void World::RegisterActor(Actor* actor)
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

void World::RegisterModel(Model* model)
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

void World::RegisterMaterial(Material* material)
{
#ifdef _DEBUG
    assert(material != nullptr);
#endif
    m_Materials.push_back(material);
}

const Texture2dDict& World::GetTexture2ds() const
{
    return m_Texture2ds;
}

void World::RegisterTexture2d(const std::string& path, Texture2d* tex2d)
{
#ifdef _DEBUG
    assert(tex2d);
    assert(m_Texture2ds.find(path) == m_Texture2ds.cend()); //no repeat registration!
#endif
    m_Texture2ds.insert({ path, tex2d });
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

std::vector<Material*> World::LoadDefaultMaterialsForModel(Model * model)
{
    std::vector<Material*> out;
    Assimp::Importer import;
    const aiScene* scene = import.ReadFile(model->GetRawPath(), aiProcess_Triangulate | aiProcess_FlipUVs);
    if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        fprintf(stderr, "assimp error: %s\n", import.GetErrorString());
        return out;
    }
    
    for (uint32_t i = 0; i < scene->mNumMaterials; i++)
    {
        Material* curr_material = new Material();
        uint32_t albedo_num = scene->mMaterials[i]->GetTextureCount(aiTextureType_DIFFUSE);
//#ifdef _DEBUG
//        assert(albedo_num > 0); //temporary
//#endif
        if (albedo_num > 0)
        {
            aiString albedo_path;
            scene->mMaterials[i]->GetTexture(aiTextureType::aiTextureType_DIFFUSE, 0, &albedo_path);
#ifdef _DEBUG
            assert(albedo_path.C_Str());
#endif


            std::string albedo_path_str(albedo_path.C_Str());
            for (size_t i = 0; i < albedo_path_str.length(); i++)
            {
                if (albedo_path_str[i] == '\\')
                    albedo_path_str[i] = '/';
            }
            const char* model_path = model->GetRawPath();
            const char* directory = strrchr(model_path, '/');
            if (directory)
                albedo_path_str.insert(0, model_path, 0, directory - model_path + 1);

            Texture2d* curr_tex2d = nullptr;
            auto tex_iter = m_Texture2ds.find(albedo_path_str);
            if (tex_iter == m_Texture2ds.cend())
            {
                curr_tex2d = new Texture2d(albedo_path_str.c_str());
                RegisterTexture2d(albedo_path_str, curr_tex2d);
            }
            else
            {
                curr_tex2d = tex_iter->second;
            }

            curr_material->AddTexture(curr_tex2d, "TexAlbedo");
        }

        RegisterMaterial(curr_material);
        out.push_back(curr_material);
    }

    
    return out;
}

int World::GetKey(int key)
{
    return glfwGetKey(m_Window, key);
}