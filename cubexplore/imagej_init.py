"""Lazy loader for pyimagej (Fiji mode)."""

import imagej

def load_imagej():
    """Initialize and return a pyimagej instance."""
    print('Initializing pyimagej in Fiji mode. Please be patient, this might take a while...')
    return imagej.init('sc.fiji:fiji')
