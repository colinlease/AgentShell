"""Personal GL application package."""


def run() -> None:
    from app.workspace_apps.personal_gl.app import run as _run

    _run()


from app.workspace_apps.personal_gl.personal_gl_app import PersonalGLApp


__all__ = ["run", "PersonalGLApp"]
