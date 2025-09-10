<<<<<<< HEAD
import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Returns a detailed error message including the filename, line number, and error message.
    """
    exc_type, exc_obj, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
    else:
        file_name = "<unknown file>"
        line_no = "<unknown line>"
    error_message = (
        f"Error occurred in python script [{file_name}] "
        f"at line [{line_no}]: {str(error)}"
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logging.error(self.error_message)  # Log the error when exception is created

    def __str__(self):
        return self.error_message
=======
import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Returns a detailed error message including the filename, line number, and error message.
    """
    exc_type, exc_obj, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
    else:
        file_name = "<unknown file>"
        line_no = "<unknown line>"
    error_message = (
        f"Error occurred in python script [{file_name}] "
        f"at line [{line_no}]: {str(error)}"
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logging.error(self.error_message)  # Log the error when exception is created

    def __str__(self):
        return self.error_message
>>>>>>> 568bd63 (New Commits)
