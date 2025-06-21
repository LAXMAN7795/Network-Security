import sys
from networkSecurity.logging import logger

class NetworkSecurityException(Exception):
    """Base class for all network security exceptions."""
    def __init__(self, error_messages, eroor_details:sys):
        self.error_messages = error_messages
        _,_,exc_tb = eroor_details.exc_info() # Extracting the traceback details

        self.line_number = exc_tb.tb_lineno # Getting the line number where the exception occurred
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"[{self.error_messages}] occurred in file [{self.file_name}] at line [{self.line_number}]"
        
