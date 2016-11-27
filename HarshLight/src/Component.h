#pragma once

class Component
{

    friend class Actor;

public:
	Component() { }
    virtual ~Component() { }
    virtual void Start() = 0;
    virtual void Update(float dt) = 0;

    inline void MarkStarted()
    { m_Started = true; }

    inline bool IsStarted()
    { return m_Started; }

    inline Actor* GetActor() const
    { return m_Actor; }

protected:
    bool m_Started = false;
    Actor* m_Actor;
};

