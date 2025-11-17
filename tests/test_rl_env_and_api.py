import numpy as np
from src.rl.reco_env import RecoEnv
import numpy as np
from src.rl.reco_env import RecoEnv


def test_env_reset_and_step():
    embs = np.random.randn(5, 16).astype(np.float32)
    news_ids = [f"N{i}" for i in range(5)]
    env = RecoEnv(embs, news_ids)
    obs, _ = env.reset()
    assert obs.shape[0] == embs.shape[1] * 2
    a = env.action_space.sample()
    obs2, reward, done, _, info = env.step(a)
    assert isinstance(reward, float)


def test_fastapi_routes():
    from fastapi.testclient import TestClient
    from api.server import app

    client = TestClient(app)
    # health endpoints
    r = client.get("/recommend/someuser?n=1")
    assert r.status_code == 200
    r2 = client.get("/user/someuser/history")
    assert r2.status_code in (200, 500)
