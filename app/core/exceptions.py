class AppBaseError(Exception):
    message = "An unexpected error occurred"

    def __init__(self, message: str = None):
        self.message = message or self.message
        super().__init__(self.message)


class FileTooLargeError(AppBaseError):
    """Raised when an uploaded file exceeds the configured size limit."""


class UnsupportedFileFormatError(AppBaseError):
    """Raised when uploaded content cannot be decoded as a supported format."""


class DocumentParseError(AppBaseError):
    """Raised when a supported document cannot be parsed into text."""


class ObjectNotFoundError(AppBaseError):
    """Raised when object is not found."""


class TaskNotFoundError(ObjectNotFoundError):
    """Raised when a request references an unknown uploaded document."""


class DocumentNotFoundError(ObjectNotFoundError):
    """Raised when a request references an unknown uploaded document."""


class AgentExecutionError(AppBaseError):
    """Raised when the agent fails or returns an invalid response."""
