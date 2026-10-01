"""Exception hierarchy. Everything costless raises on purpose derives from CostlessError."""


class CostlessError(Exception):
    """Base class for all costless errors."""


class ConfigError(CostlessError):
    """The configuration file is missing, malformed or inconsistent."""


class DatasetError(CostlessError):
    """A dataset file is missing, malformed or contains invalid cases."""


class TargetError(CostlessError):
    """The system under test failed to produce an output."""


class ReplayMissError(CostlessError):
    """A replay provider was asked for a request it has no recording of."""


class ResultsError(CostlessError):
    """A results file cannot be read or has an incompatible schema."""


class ProviderError(CostlessError):
    """A model provider call failed after retries."""
