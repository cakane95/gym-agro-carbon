import asyncio
import numpy as np
from pathlib import Path
from gym_agro_carbon.envs.gama_env import GamaAgroCarbonEnv
from gym_agro_carbon.envs.grid import EnvSpec
from gym_agro_carbon.models.context import ContextSpec, ContextEncoder
from gym_agro_carbon.models.reward import RewardModel, RewardSpec

async def test_gama_handshake():
    print("\n" + "="*50)
    print("🚀 STARTING ASYNC GAMA COUPLING TEST")
    print("="*50)

    project_root = Path(__file__).resolve().parent.parent
    gaml_file = project_root / "gama_model" / "models" / "EcoSysML" / "main.gaml"

    env_spec = EnvSpec(H=10, W=10, T=5, S=8, M=7, alpha=0.5)
    context_spec = ContextSpec(S=8, M=7)
    encoder = ContextEncoder(context_spec)
    reward_model = RewardModel(RewardSpec(context_spec, alpha=0.5), encoder)

    # L'environnement utilise le wrapper qui contient le GamaSyncClient
    env = GamaAgroCarbonEnv(
        env_spec, context_spec, reward_model, 
        gaml_file_path=str(gaml_file)
    )

    try:
        print("🔄 Resetting Environment...")
        # reset() appelle load_experiment() en interne
        obs, info = env.reset(seed=42)
        print(f"✅ Reset successful. ID Exp: {info.get('exp_id')}")

        print("\n🏃 Running 10 Steps...")
        for t in range(10):
            actions = np.random.randint(0, 4, size=(10, 10))
            obs, reward, term, trunc, info = env.step(actions)
            
            print(f"  Step {t+1}: Reward = {reward:.4f}, Mean SOC = {np.mean(info['soc_map']):.2f}")

    finally:
        env.close()
        print("="*50)

if __name__ == "__main__":
    # Fournit la boucle d'événements requise par le client GAMA
    asyncio.run(test_gama_handshake())