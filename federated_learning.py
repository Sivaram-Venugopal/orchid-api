import numpy as np
import logging
import hashlib
import hmac

logger = logging.getLogger(__name__)

class FederatedCoordinator:
    """
    Manages secure Federated Averaging (FedAvg) aggregation of local RL policy updates.
    Ensures operator data privacy while training a shared global avoidance model.
    """
    def __init__(self):
        self.submissions = {}
        # Pre-registered operator public commitment keys
        self.operator_secrets = {
            "operator_spacex": b"spacex_secret_handshake_key_101",
            "operator_oneweb": b"oneweb_secret_handshake_key_202",
            "operator_isro": b"isro_secret_handshake_key_303"
        }

    def verify_operator_signature(self, operator_id: str, payload_hash: str, signature: str) -> bool:
        """Verifies HMAC signature of the operator's payload to prevent model poisoning."""
        secret = self.operator_secrets.get(operator_id)
        if not secret:
            logger.warning(f"Operator {operator_id} is not registered in the Federated group.")
            return False
            
        try:
            expected_sig = hmac.new(secret, payload_hash.encode(), hashlib.sha256).hexdigest()
            return hmac.compare_digest(expected_sig, signature)
        except Exception as e:
            logger.error(f"Error validating operator signature: {e}")
            return False

    def submit_local_weights(self, operator_id: str, weights: list, sample_count: int, signature: str) -> dict:
        """
        Submits local model weights to the coordinator queue.
        Calculates payload hash and verifies operator signature before queuing.
        """
        # Create hash of the weights array representation to verify signature integrity
        weights_str = str(weights)
        payload_hash = hashlib.sha256(weights_str.encode()).hexdigest()
        
        if not self.verify_operator_signature(operator_id, payload_hash, signature):
            return {"status": "failed", "message": "Authentication signature mismatch"}
            
        # Queue the submission
        self.submissions[operator_id] = {
            "weights": [np.array(w, dtype=np.float32) for w in weights],
            "sample_count": sample_count
        }
        logger.info(f"Queued federated update from {operator_id} containing {sample_count} training samples.")
        
        return {
            "status": "success", 
            "message": f"Weights registered successfully. Queue size: {len(self.submissions)}"
        }

    def aggregate_weights(self) -> list:
        """
        Computes Federated Averaging (FedAvg) across all queued submissions:
        W_global = sum( (samples_i / total_samples) * W_i )
        """
        if len(self.submissions) < 2:
            logger.warning("Aggregate aborted: Not enough operator submissions in the queue (minimum 2).")
            return []
            
        total_samples = sum(s["sample_count"] for s in self.submissions.values())
        if total_samples == 0:
            return []
            
        global_weights = []
        
        # Get shape dimensions from the first submission
        first_sub = list(self.submissions.values())[0]["weights"]
        for layer_idx in range(len(first_sub)):
            layer_sum = np.zeros_like(first_sub[layer_idx])
            
            for sub in self.submissions.values():
                weight_factor = sub["sample_count"] / total_samples
                layer_sum += sub["weights"][layer_idx] * weight_factor
                
            global_weights.append(layer_sum.tolist())
            
        logger.info(f"Successfully aggregated model weights across {len(self.submissions)} operators. Total samples processed: {total_samples}.")
        self.submissions.clear() # Reset queue for next round
        return global_weights
