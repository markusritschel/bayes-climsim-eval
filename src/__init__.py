# !/usr/bin/env python3
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-04-17
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import inspect
import logging
import os
import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

__version__ = '0.1.0'

# Make some of the basic directories globally available in your environment
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
LOG_DIR  = BASE_DIR / 'logs'
PLOT_DIR = BASE_DIR / 'reports/figures'
jupyter_startup_script = BASE_DIR / 'notebooks/jupyter_startup.ipy'

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)

sys.path.append(str(BASE_DIR/"scripts"))


def setup_logger(level=None, logfile=True):
    """Define a logger setup with
    - 1x fileHandler: writing log files to LOG_DIR (logfile can be boolean or a file path)
    - 1x streamHandler: streaming logs to terminal

    Parameters
    ----------
    level : logLevel / str
        The log level (according to the logging convention). Can be either a string or a 
        loggin.loglevel instance
    logfile : bool / str
        If True, the log file will be placed in LOG_DIR and named after the calling script (default)
        If logfile is a string, it will be interpreted as a file path (the parent directory must exist)
    
    Return
    ------
    logger
        The logger instance
    """
    caller_file = inspect.stack()[-1].filename
    caller_filename = Path(caller_file).stem

    if not level:
        level = os.getenv('LOGLEVEL', 'INFO').upper()

    # set up new logger and set level
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s: '
                                  '[%(levelname)s] '
                                  # '(%(name)s): '
                                  '(%(name)s:#%(lineno)d): '
                                  '%(message)s'
                                  , datefmt='%Y-%m-%d %H:%M:%S')
    if logfile:
        if isinstance(logfile, bool):
            logfile = LOG_DIR / f'{caller_filename}_{os.getpid()}.log'
        elif isinstance(logfile, str):
            logdir = Path(logfile).parents[0]
            if not logdir.exists():
                raise IOError(f'Log directory {logdir} does not exist.')
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
        logger.logfile = logfile.as_posix()
    else:
        logger.logfile = None

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

#     logger.info("="*(17+len(caller_file)))
#     logger.info("Calling routine: %s", caller_file)
#     if logfile:
#         logger.info("Log file: %s", logfile)
#     logger.info("-"*(17+len(caller_file))+"\n")

    return logger



welcome = """
████████╗██╗████████╗██╗     ███████╗
╚══██╔══╝██║╚══██╔══╝██║     ██╔════╝
   ██║   ██║   ██║   ██║     █████╗  
   ██║   ██║   ██║   ██║     ██╔══╝  
   ██║   ██║   ██║   ███████╗███████╗
   ╚═╝   ╚═╝   ╚═╝   ╚══════╝╚══════╝
          Some subtitle ☃
"""
# https://patorjk.com/software/taag/ with "ANSI Shadow" font

# print(pyfiglet.figlet_format("My title", font="slant") + "\n Some subtitle")


if __name__ == '__main__':
    print(welcome)
