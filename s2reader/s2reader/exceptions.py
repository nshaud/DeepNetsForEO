"""Errors and Warnings."""


class S2ReaderIOError(IOError):
    """Raised if an expected file cannot be found."""


class S2ReaderMetadataError(Exception):
    """Raised if metadata structure is not as expected."""
