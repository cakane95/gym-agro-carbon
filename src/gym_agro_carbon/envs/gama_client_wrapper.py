"""
GamaClientWrapper: Version finale et épurée.
Compatible avec gama-client >= 1.2.1 (via gama-gymnasium).
"""
import logging
from typing import Any, Dict, List, Optional

from gama_client.sync_client import GamaSyncClient
from gama_client.message_types import MessageTypes

from .exceptions import GamaConnectionError, GamaCommandError

logger = logging.getLogger(__name__)

class GamaClientWrapper:
    """
    Wrapper de haut niveau autour de GamaSyncClient.
    Gère la communication directe avec l'agent d'interface GAML.
    """

    def __init__(self, ip_address: str = "localhost", port: int = 6868):
        self.ip_address = ip_address
        self.port = port
        self.client = None
        self.experiment_id = None
        
        # Le package gama-client s'occupe de l'event loop en interne
        self._connect()

    def _connect(self):
        """Établit la connexion avec les paramètres positionnels requis."""
        try:
            # Signature standard: url, port, async_handler, message_handler
            self.client = GamaSyncClient(
                self.ip_address, 
                self.port, 
                self._async_command_handler,
                self._server_message_handler
            )
            # Appel minimal de connect sans keyword arguments problématiques
            self.client.connect(False) 
            logger.info(f"Connecté à GAMA sur ws://{self.ip_address}:{self.port}")
        except Exception as e:
            raise GamaConnectionError(f"Échec de la connexion au serveur GAMA : {e}")

    async def _async_command_handler(self, message: dict):
        pass

    async def _server_message_handler(self, message: dict):
        pass

    def load_experiment(self, gaml_path: str, experiment_name: str, 
                        parameters: list = None) -> str:
        """Charge le modèle et démarre l'expérience."""
        try:
            response = self.client.load(
                gaml_path, 
                experiment_name, 
                console=False, 
                runtime=True, 
                parameters=parameters or []
            )
            
            if response.get("type") != MessageTypes.CommandExecutedSuccessfully.value:
                raise GamaCommandError(f"Erreur de chargement : {response}")
            
            self.experiment_id = response.get("content")
            
            # Lancement immédiat
            self.client.play(self.experiment_id)
            return self.experiment_id
            
        except Exception as e:
            raise GamaCommandError(f"Erreur lors du load/play : {e}")

    def execute_step(self, action_list: List[int]) -> Dict:
        """
        Version stable : utilise l'assignation directe avec point-virgule 
        et l'appel natif au moteur de simulation GAMA.
        """
        # 1. Injection de l'action (le ";" est crucial ici pour fermer l'instruction GAML)
        self._execute_expression(f"GymAgent[0].next_action <- {action_list};")
        
        # 2. Avancement d'un cycle complet de simulation
        # Cela déclenche tous les réflexes GAML (biophysique, etc.)
        response = self.client.step(self.experiment_id, True)
        
        # 3. Récupération des données mises à jour
        raw_data = self._execute_expression("GymAgent[0].data")
        return raw_data

    def _execute_expression(self, expression: str):
        """Exécute une expression GAML et retourne le contenu."""
        if not self.experiment_id:
            raise GamaCommandError("Aucune expérience active.")
            
        response = self.client.expression(self.experiment_id, expression)
        
        if response.get("type") != MessageTypes.CommandExecutedSuccessfully.value:
            raise GamaCommandError(f"L'expression '{expression}' a échoué : {response}")
            
        return response.get("content")

    def close(self):
        """Fermeture propre de la session (alias de stop pour compatibilité env)."""
        if self.client and self.experiment_id:
            try:
                # On tente d'arrêter l'expérience sur le serveur
                self.client.stop(self.experiment_id)
                # On ferme la connexion websocket
                self.client.close_connection()
            except Exception as e:
                logger.warning(f"Erreur lors de la fermeture : {e}")
            finally:
                self.client = None
                self.experiment_id = None