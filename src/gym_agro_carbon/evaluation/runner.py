import numpy as np

def run_agent(env, agent, reward_model, regret_tracker, T, episode_idx=0, seed=None):
    """
    Exécute un épisode complet de manière SYNCHRONE.
    """
    # 1. Initialisation
    obs, info = env.reset(seed=seed, options={"episode_id": episode_idx})
    agent.reset()
    regret_tracker.reset()
    
    trajectory = []
    
    for t in range(T):
        # 2. Observation et Décision
        # obs est déjà une matrice (10, 10) grâce au reshape dans ton step
        context_ids = obs 
        actions = agent.select_actions(context_ids)
        
        # 3. Step dans GAMA
        # rewards_raw est un float (somme totale), on ne l'utilise pas pour la VNA par cellule
        next_obs, rewards_raw, terminated, truncated, info = env.step(actions)
        
        # 4. Calcul du Regret (Pseudo-regret basé sur les moyennes mu)
        step_regret = regret_tracker.update(
            t=t, 
            obs_context_ids=context_ids, 
            actions_grid=actions
        )
        
        # 5. Transformation VNA
        # On initialise learning_rewards sur la forme de context_ids (10x10)
        # pour éviter l'IndexError lié au scalaire rewards_raw
        learning_rewards = np.zeros(context_ids.shape, dtype=np.float32)
        
        # Récupération des maps NumPy déjà préparées par ton environnement
        soc_data = info["soc_map"]
        yield_data = info["yield_map"]

        # Itération sur la grille pour calculer la récompense d'apprentissage VNA
        for i in range(context_ids.shape[0]):
            for j in range(context_ids.shape[1]):
                ctx_id = context_ids[i, j]
                s, tau = reward_model.encoder.from_id(ctx_id)
                
                # Calcul de la VNA ou neutralisation (0.0) selon l'âge tau
                learning_rewards[i, j] = reward_model.get_learning_reward(
                    action=actions[i, j],
                    s=s,
                    tau=tau,
                    real_delta_soc=soc_data[i, j],
                    real_yield=yield_data[i, j]
                )
        
        # 6. Mise à jour de l'agent avec la grille de récompenses VNA
        agent.update(context_ids, actions, learning_rewards)
        
        # 7. Logging de la trajectoire
        trajectory.append({
            "step": t,
            "mean_delta_soc": np.mean(soc_data),
            "mean_yield": np.mean(yield_data),
            "step_regret": step_regret,
            "mean_action": np.mean(actions), 
            "action_counts": np.bincount(actions.flatten(), minlength=4).tolist()
        })
        
        obs = next_obs
        if terminated or truncated:
            break
            
    return trajectory, regret_tracker.cumulative_regret