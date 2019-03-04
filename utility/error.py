# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 16:37:46 2018

@author: Morgan.Li
"""

class InvalidDenomError(Exception):
        def __init__(self):
            self.message = "Invalid denominator. Divide Zero!"
        def __str__(self):
            return repr(self.InvalidDenomError)