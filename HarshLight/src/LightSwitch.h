#pragma once
#include "Component.h"

class LightSwitch : public Component
{
public:
	LightSwitch() { }
	
	void Start() override;
	void Update(float dt) override;

private:

	float m_CurrentPitchDeg;
	float m_CurrentYawDeg;
};