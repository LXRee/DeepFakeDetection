class BaseError(Exception):
    """
    Base error class for others
    """
    pass


class NoFrames(BaseError):
    """
    Call it if there are no frames in video (video missing)
    """
    pass


class NoFaces(BaseError):
    """
    Call it if there are no faces in video (there are frames but without faces)
    """
