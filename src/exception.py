import sys
import traceback
from datetime import datetime
from typing import Optional, Dict, Any
from src.logger import logging

class CustomException(Exception):
    """
    Enhanced custom exception class for ML pipeline error handling and debugging.
    Provides detailed error tracking, context preservation, and formatted output.
    """
    
    def __init__(
        self, 
        error_message: str, 
        error_detail: sys,
        context: Optional[Dict[str, Any]] = None,
        should_log: bool = True
    ):
        """
        Initialize the CustomException with comprehensive error information.
        
        Args:
            error_message: The original error message
            error_detail: System information about the error
            context: Optional dictionary containing relevant context (e.g., pipeline stage, data state)
            should_log: Whether to automatically log the error (default: True)
        """
        super().__init__(error_message)
        
        self.timestamp = datetime.now()
        self.context = context or {}
        self.error_info = self._collect_error_info(error_message, error_detail)
        
        if should_log:
            self._log_error()

    def _collect_error_info(self, error_message: str, error_detail: sys) -> Dict[str, Any]:
        """
        Collect comprehensive error information including stack trace and context.
        
        Args:
            error_message: The original error message
            error_detail: System information about the error
            
        Returns:
            Dict containing detailed error information
        """
        exc_type, exc_value, exc_tb = error_detail.exc_info()
        
        # Get full stack trace
        stack_summary = traceback.extract_tb(exc_tb)
        
        # Get the immediate failure point
        failing_frame = exc_tb.tb_frame
        failing_file = failing_frame.f_code.co_filename
        failing_func = failing_frame.f_code.co_name
        failing_line = exc_tb.tb_lineno
        
        # Collect local variables from the failing frame
        local_vars = {
            key: str(value) 
            for key, value in failing_frame.f_locals.items()
            if not key.startswith('_')  # Exclude private variables
        }

        return {
            'timestamp': self.timestamp.isoformat(),
            'error_type': exc_type.__name__ if exc_type else 'Unknown',
            'error_message': str(error_message),
            'failing_file': failing_file,
            'failing_function': failing_func,
            'failing_line': failing_line,
            'stack_trace': [
                {
                    'filename': frame.filename,
                    'lineno': frame.lineno,
                    'function': frame.name,
                    'code': frame.line
                }
                for frame in stack_summary
            ],
            'local_variables': local_vars,
            'context': self.context
        }

    def _log_error(self) -> None:
        """
        Log the error with different severity levels based on context.
        """
        message = self.format_error(include_locals=True)
        
        # Determine logging level based on context or error type
        if self.context.get('critical', False):
            logging.critical(message)
        else:
            logging.error(message)

    def format_error(self, include_locals: bool = False) -> str:
        """
        Format the error information into a readable string.
        
        Args:
            include_locals: Whether to include local variables in the output
            
        Returns:
            Formatted error message string
        """
        info = self.error_info
        
        message = [
            "\n=== ML Pipeline Error Report ===",
            f"Timestamp: {info['timestamp']}",
            f"Error Type: {info['error_type']}",
            f"Error Message: {info['error_message']}",
            f"\nFailure Location:",
            f"→ File: {info['failing_file']}",
            f"→ Function: {info['failing_function']}",
            f"→ Line: {info['failing_line']}",
            "\nStack Trace:"
        ]
        
        # Add stack trace
        for frame in info['stack_trace']:
            message.append(
                f"  • {frame['filename']}:{frame['lineno']} "
                f"in {frame['function']}\n    {frame['code']}"
            )
        
        # Add context if available
        if info['context']:
            message.extend([
                "\nContext:",
                *[f"→ {k}: {v}" for k, v in info['context'].items()]
            ])
        
        # Add local variables if requested
        if include_locals and info['local_variables']:
            message.extend([
                "\nLocal Variables:",
                *[f"→ {k} = {v}" for k, v in info['local_variables'].items()]
            ])
        
        message.append("=" * 30)
        
        return "\n".join(message)

    def __str__(self) -> str:
        """
        String representation of the exception.
        
        Returns:
            Formatted error message without local variables
        """
        return self.format_error(include_locals=False)