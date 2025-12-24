#  Copyright 2025 Sony Semiconductor Solutions, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import os
import shutil

from model_compression_toolkit.logger import Logger, set_log_folder
from mct_quantizers import logger as mct_quantizers_logger


def test_logger():

    set_log_folder("./test_debug", level=Logger.DEBUG)

    Logger.debug("DEBUG message 1")
    Logger.info("INFO message 1")
    Logger.warning("WARNING message 1")
    Logger.error("ERROR message 1")

    mct_quantizers_logger.Logger.debug("DEBUG message 2")
    mct_quantizers_logger.Logger.info("INFO message 2")
    mct_quantizers_logger.Logger.warning("WARNING message 2")

    # Verify MCT log file content
    mct_log_file = os.path.join(Logger.LOG_PATH, "mct_log.log")
    with open(mct_log_file, "r") as f:
        mct_log_content = f.read()
    
    assert "DEBUG message 1" in mct_log_content
    assert "INFO message 1" in mct_log_content
    assert "WARNING message 1" in mct_log_content
    assert "ERROR message 1" in mct_log_content

    # Verify MCTQ log file content (uses _MCTQ suffix folder)
    mctq_log_file = os.path.join(mct_quantizers_logger.Logger.LOG_PATH, "mct_log.log")
    with open(mctq_log_file, "r") as f:
        mctq_log_content = f.read()
    
    assert "DEBUG message 2" in mctq_log_content
    assert "INFO message 2" in mctq_log_content
    assert "WARNING message 2" in mctq_log_content

    # Cleanup
    Logger.shutdown()
    mct_quantizers_logger.Logger.shutdown()
    if os.path.exists("./test_debug"):
        shutil.rmtree("./test_debug")
    if os.path.exists("./test_debug_MCTQ"):
        shutil.rmtree("./test_debug_MCTQ")