class ServiceError(Exception):
    """Base exception for service-layer failures."""


class FileTooLargeError(ServiceError):
    """Raised when an uploaded file exceeds the configured size limit."""


class UnsupportedFileFormatError(ServiceError):
    """Raised when uploaded content cannot be decoded as a supported format."""


class DocumentParseError(ServiceError):
    """Raised when a supported document cannot be parsed into text."""


class DocumentNotFoundError(ServiceError):
    """Raised when a request references an unknown uploaded document."""


class AgentExecutionError(ServiceError):
    """Raised when the agent fails or returns an invalid response."""
