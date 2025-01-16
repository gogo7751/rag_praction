from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import root
from services.demo import initialize_demo_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

initialize_demo_data()
# 載入路由
app.include_router(root.router)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
