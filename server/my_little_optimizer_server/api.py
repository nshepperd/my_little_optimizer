def create_app(settings=None):
    from fastapi import FastAPI
    from contextlib import asynccontextmanager
    from my_little_optimizer_server.server.manager import SweepManager

    if settings is None:
        from .config import settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.manager = SweepManager(db_path=settings.db_path)
        yield
        app.state.manager.close()

    app = FastAPI(
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )


    # Import and register routes
    from .routes import register_routes
    register_routes(app)
    
    return app