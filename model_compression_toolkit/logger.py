# Copyright 2021 Sony Semiconductor Solutions, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import logging
import os
from datetime import datetime
from pathlib import Path
import importlib.util
import inspect
import sys

LOGGER_NAME = 'Model Compression Toolkit'
LOG_FORMAT = '%(caller_module)s - %(caller_filename)s:%(caller_lineno)d - %(message)s'


class CallerFormatter(logging.Formatter):
    """Custom formatter to retrieve caller's file information"""
    def format(self, record):
        # Get the caller's frame by skipping the Logger class to find the original code
        frame = inspect.currentframe()
        
        # Find the actual caller by traversing beyond the logging system and Logger class
        while frame:
            # フレーム情報を表示
            # print(f"Inspecting frame: {frame.f_code.co_filename}, line {frame.f_lineno}")
            
            frame_info = inspect.getframeinfo(frame)
            # Find a frame that is not inside the Logger class
            if 'logger.py' not in frame_info.filename and 'logging' not in frame_info.filename:
                caller_frame = frame
                
                # Get package information from the caller frame
                caller_module = inspect.getmodule(caller_frame)
                if caller_module:
                    record.caller_module = caller_module.__package__ if hasattr(caller_module, '__package__') else None
                else:
                    # Fallback: extract from file path
                    file_path = caller_frame.f_code.co_filename
                    parts = file_path.replace(os.sep, '/').split('/')
                    record.caller_module = parts[-2] if len(parts) > 1 else 'unknown'                    
                break
            frame = frame.f_back

        record.caller_filename = os.path.basename(caller_frame.f_code.co_filename)
        record.caller_lineno = caller_frame.f_lineno
        return super().format(record)

class Logger:
    # Logger has levels of verbosity.
    LOG_PATH = None
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    @staticmethod
    def __check_path_create_dir(log_path: str):
        """
        Create a path if not exist. Otherwise, do nothing.
        Args:
            log_path: Path to create or verify that exists.

        """

        if not os.path.exists(log_path):
            Path(log_path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def set_logger_level(log_level=logging.INFO):
        """
        Set log level to determine the logger verbosity.
        Args:
            log_level: Level of verbosity to set for the logger.

        """

        logger = Logger.get_logger()
        logger.setLevel(log_level)

    @staticmethod
    def set_handler_level(log_level=logging.INFO):
        """
        Set log level for all handlers attached to the logger.
        Args:
            log_level: Level of verbosity to set for the handlers.

        """

        logger = Logger.get_logger()
        for handler in logger.handlers:
            handler.setLevel(log_level)

    @staticmethod
    def get_logger():
        """
        Returns: An instance of the logger.
        """
        return logging.getLogger(LOGGER_NAME)
        
    @staticmethod
    def set_stream_handler():
        """
        Add a StreamHandler to output logs to the console (stdout).
        """
        logger = Logger.get_logger()
        
        # Check if StreamHandler already exists
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                return
        
        # Add StreamHandler
        sh = logging.StreamHandler()
        formatter = CallerFormatter(LOG_FORMAT)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    @staticmethod
    def set_log_file(log_folder: str = None):
        """
        Setting the logger log file path. The method gets the folder for the log file.
        In that folder, it creates a log file according to the timestamp.
        Args:
            log_folder: Folder path to hold the log file.

        """

        logger = Logger.get_logger()

        ts = datetime.now(tz=None).strftime("%d%m%Y_%H%M%S")

        if log_folder is None:
            Logger.LOG_PATH = os.path.join(os.environ.get('LOG_PATH', os.getcwd()), f"logs_{ts}")
        else:
            Logger.LOG_PATH = os.path.join(log_folder, f"logs_{ts}")
        log_name = os.path.join(Logger.LOG_PATH, f'mct_log.log')

        Logger.__check_path_create_dir(Logger.LOG_PATH)

        fh = logging.FileHandler(log_name)
        formatter = CallerFormatter(LOG_FORMAT)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        print(f'log file is in {log_name}')

    @staticmethod
    def shutdown():
        """
        An orderly command to shutdown by flushing and closing all logging handlers.

        """
        Logger.LOG_PATH = None
        logging.shutdown()

    ########################################
    # Delegating methods to wrapped logger
    ########################################

    @staticmethod
    def critical(msg: str):
        """
        Log a message at 'critical' severity and raise an exception.
        Args:
            msg: Message to log.

        """
        Logger.get_logger().critical(msg)
        raise Exception(msg)

    @staticmethod
    def exception(msg: str):
        """
        Log a message at 'exception' severity and raise an exception.
        Args:
            msg: Message to log.

        """
        Logger.get_logger().exception(msg)
        raise Exception(msg)

    @staticmethod
    def debug(msg: str):
        """
        Log a message at 'debug' severity.

        Args:
            msg: Message to log.

        """
        Logger.get_logger().debug(msg)

    @staticmethod
    def info(msg: str):
        """
        Log a message at 'info' severity.

        Args:
            msg: Message to log.

        """
        Logger.get_logger().info(msg)

    @staticmethod
    def warning(msg: str):
        """
        Log a message at 'warning' severity.

        Args:
            msg: Message to log.

        """
        Logger.get_logger().warning(msg)

    @staticmethod
    def error(msg: str):
        """
        Log a message at 'error' severity and raise an exception.

        Args:
            msg: Message to log.

        """
        Logger.get_logger().error(msg)


def set_log_folder(folder: str, level: int = logging.INFO):
    """
    Set a directory path for saving a log file.

    Args:
        folder: Folder path to save the log file.
        level: Level of verbosity to set to the logger and handlers.

    Note:
        This is a convenience function that calls multiple Logger methods
        to set up logging.

        Don't use Python's original logger.
    """

    Logger.set_stream_handler()
    Logger.set_log_file(folder)
    Logger.set_logger_level(level)
    Logger.set_handler_level(level)
