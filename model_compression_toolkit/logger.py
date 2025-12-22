# Copyright 2025 Sony Semiconductor Solutions, Inc. All rights reserved.
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
from typing import Optional


LOGGER_NAME = 'Model Compression Toolkit'


class Logger:
    # Logger has levels of verbosity.
    LOG_PATH = None
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    @staticmethod
    def __check_path_create_dir(log_path: str) -> None:
        """
        Create a path if not exist. Otherwise, do nothing.

        Args:
            log_path (str): Path to create or verify that exists.
        """

        if not os.path.exists(log_path):
            Path(log_path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def set_logger_level(log_level: int = logging.INFO) -> None:
        """
        Set log level to determine the logger verbosity.

        Args:
            log_level (int): Level of verbosity to set for the logger.
        """

        logger = Logger.get_logger()
        logger.setLevel(log_level)

    @staticmethod
    def set_handler_level(log_level: int = logging.INFO) -> None:
        """
        Set log level for all handlers attached to the logger.

        Args:
            log_level (int): Level of verbosity to set for the handlers.
        """

        logger = Logger.get_logger()
        for handler in logger.handlers:
            handler.setLevel(log_level)

    @staticmethod
    def get_logger() -> logging.Logger:
        """
        Returns: An instance of the logger.
        """
        return logging.getLogger(LOGGER_NAME)

    @staticmethod
    def set_stream_handler() -> None:
        """
        Add a StreamHandler to output logs to the console (stdout).
        """
        logger = Logger.get_logger()
        
        # Check if StreamHandler already exists
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                return
        
        # Add StreamHandler
        sh = logging.StreamHandler()
        logger.addHandler(sh)

    @staticmethod
    def set_log_file(log_folder: Optional[str] = None) -> None:
        """
        Setting the logger log file path. The method gets the folder for the log file.
        In that folder, it creates a log file according to the timestamp.

        Args:
            log_folder (Optional[str]): Folder path to hold the log file.
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
        logger.addHandler(fh)

        print(f'log file is in {log_name}')

    @staticmethod
    def shutdown() -> None:
        """
        An orderly command to shutdown by flushing and closing all logging handlers.
        """
        Logger.LOG_PATH = None
        logging.shutdown()

    ########################################
    # Delegating methods to wrapped logger
    ########################################

    @staticmethod
    def critical(msg: str) -> None:
        """
        Log a message at 'critical' severity and raise an exception.

        Args:
            msg (str): Message to log.
        """
        Logger.get_logger().critical(msg)
        raise Exception(msg)

    @staticmethod
    def exception(msg: str) -> None:
        """
        Log a message at 'exception' severity and raise an exception.

        Args:
            msg (str): Message to log.
        """
        Logger.get_logger().exception(msg)
        raise Exception(msg)

    @staticmethod
    def debug(msg: str) -> None:
        """
        Log a message at 'debug' severity.

        Args:
            msg (str): Message to log.
        """
        Logger.get_logger().debug(msg)

    @staticmethod
    def info(msg: str) -> None:
        """
        Log a message at 'info' severity.

        Args:
            msg (str): Message to log.
        """
        Logger.get_logger().info(msg)

    @staticmethod
    def warning(msg: str) -> None:
        """
        Log a message at 'warning' severity.

        Args:
            msg (str): Message to log.
        """
        Logger.get_logger().warning(msg)

    @staticmethod
    def error(msg: str) -> None:
        """
        Log a message at 'error' severity and raise an exception.

        Args:
            msg (str): Message to log.
        """
        Logger.get_logger().error(msg)


def set_log_folder(folder: str, level: int = logging.INFO) -> None:
    """
    Set a directory path for saving a log file.

    Args:
        folder (str): Folder path to save the log file.
        level (int): Level of verbosity to set to the logger and handlers.

    Note:
        This is a convenience function that calls multiple Logger methods
        to set up logging.

        Don't use Python's original logger.
    """

    Logger.set_stream_handler()
    Logger.set_log_file(folder)
    Logger.set_logger_level(level)
    Logger.set_handler_level(level)

    # Create _MCTQ folder (append suffix directly without separator)
    mctq_folder = folder + "_MCTQ"
    
    # Call set_log_folder from mct-quantizers
    try:
        # Import from installed mct-quantizers package
        from mct_quantizers import logger as mct_quantizers_logger
        mct_quantizers_logger.set_log_folder(mctq_folder, level)
    except Exception as e:
        Logger.warning(f"Failed to import set_log_folder from mct-quantizers: {e}")
