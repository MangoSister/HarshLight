#pragma once

class Component
{
public:
	Component() { }
    virtual ~Component() { }
    virtual void Start() = 0;
    virtual void Update(float dt) = 0;

    inline void MarkStarted()
    { m_Started = true; }

    inline bool IsStarted()
    { return m_Started; }

private:
    bool m_Started = false;
};