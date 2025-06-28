import os
import sys
sys.path.insert(0, os.path.abspath('../visPLred/docs'))

extensions = ['myst_parser',
              'myst_nb']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
html_theme = 'alabaster'
nb_execution_mode = 'off'