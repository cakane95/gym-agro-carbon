"""
Custom exceptions for GAMA-Gymnasium integration.
"""

class GamaEnvironmentError(Exception):
    """Base exception for GAMA Environment errors."""
    pass

class GamaConnectionError(GamaEnvironmentError):
    """Raised when connection to GAMA server fails."""
    pass

class GamaCommandError(GamaEnvironmentError):
    """Raised when a GAMA command execution fails or returns an error."""
    pass