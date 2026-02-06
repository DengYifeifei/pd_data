# -*- coding: utf-8 -*-

"""
This file is part of eyelinkparser.

"""

from eyelinkparser._events import sample, fixation, saccade, blink
from eyelinkparser._eyelinkparser import EyeLinkParser
from eyelinkparser._trialprocessor import TrialProcessor
from eyelinkparser._dataprocessor import DataProcessor
__version__ = '0.17.5'


def parse(parser=EyeLinkParser, **kwdict):

    return parser(**kwdict).dm

def trial_processor(processor=TrialProcessor, **kwdict):

    return processor(**kwdict)

def data_processor(processor=DataProcessor, **kwdict):

    return processor(**kwdict)