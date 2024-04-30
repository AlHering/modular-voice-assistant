# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from typing import Union, List, Optional
import re
import subprocess
from subprocess import PIPE


def download_with_wget(download_link: str, target_path: str, continue_download: bool = True,
                       time_out: int = 10, retry: int = 3, use_torsocks: bool = False) -> bool:
    """
    Function for downloading files to specified target path.
    :param download_link: Download link.
    :param target_path: Full path to download file to.
    :param continue_download: Specifies whether download should continue to download a partly downloaded file.
    Defaults to True.
    :param time_out: Declares timeout in seconds. Defaults to 10.
    :param retry: Declares number of retries. Defaults to 3.
    :param use_torsocks: Declaration, whether to use torsocks or not.
    :return: True, if command was successful, else False.
    """
    # Attention! If command fails on Windows: Open administrative command prompt and run 'choco install wget'.
    not_finished = True
    for i in range(retry):
        if use_torsocks:
            command = "torsocks "
        else:
            command = ""
        command += "wget"
        if continue_download:
            command += " -c"
        command += " -T " + str(time_out) + ' -O "' + \
            target_path + '" "' + download_link + '"'
        if issue_cli_command(command, error_pattern=r"ERROR"):
            not_finished = False
        else:
            continue_download = True
    return not not_finished


def issue_cli_command(command: Union[str, List[str]], success_pattern: str = r'.*', error_pattern: str = r'.*') -> Optional[bool]:
    """
    Function for issuing command and getting live info.
    Taken from @https://stackoverflow.com/a/52091495 and adjusted.
    :param command: Command to issue.
    :param success_pattern: Regular expression to validate successful command run by last line. Defaults to r'.*'.
    :param error_pattern: Regular expression to validate failed command run by last line. Defaults to r'.*'.
    :return: True, if command was successful, False if command failed, None if unknown.
    """
    not_finished = True
    lines = []
    while not_finished:
        lines = []
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   stdin=PIPE,
                                   shell=True)

        while process.stdout.readable():
            line = process.stdout.readline()
            if not line:
                not_finished = False
                break
            lines.append(line.decode().strip())
    if re.match(error_pattern, lines[-1]):
        return False
    if re.match(success_pattern, lines[-1]):
        return True
    return None
