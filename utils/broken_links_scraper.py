# This file is released under MIT licence, see LICENSE file.
# Author(s):  Hind Montassif
#
# Copyright (C) 2021 Inria

import re
import requests
import sys
from urllib.request import urlopen
from colorama import Fore

log_verbose = False # Not verbose by default (only handles 404 errors) ; if set to True, the other errors types are displayed among other traces

if len(sys.argv) >= 2:
    if sys.argv[1] == "verbose":
        log_verbose = True
    else :
        print("The only supported additional argument is 'verbose'")
        exit(1)

def print_cond(cond, str):
    if cond:
        print(str)
    else:
        pass

exit_status = 0
# Get data from gudhi TDA-tutorial repo README
data  = urlopen("https://raw.githubusercontent.com/GUDHI/TDA-tutorial/master/README.md").read().decode('utf-8')

# Get all websites links (which begin with http or https)
name_regex = "[^]]+"
url_regex = "http[s]?://[^)]+"

markup_regex = '\[({0})]\(\s*({1})\s*\)'.format(name_regex, url_regex)

list_http_links = []
for match in re.findall(markup_regex, data):
    if (match[1].find('latex') == -1):
        list_http_links.append(match[1])

# Check validity of these urls
print_cond(log_verbose, Fore.CYAN + "Checking websites URLs status code ... " + Fore.RESET)

all_good = True
for url in list_http_links:
    try:
        r = requests.head(url)
        if (r.status_code != 200):
            all_good = False
            if (r.status_code == 404):
                print(Fore.RED + "{} is not working. The returned status code is {:4d}".format(url, r.status_code) + Fore.RESET)
                exit_status = 1
            else:
                print_cond(log_verbose, Fore.LIGHTYELLOW_EX + "{} may not be working. The returned status code is {:4d}".format(url, r.status_code) + Fore.RESET)
    except requests.ConnectionError:
        print_cond(log_verbose, Fore.RED + "Failed to connect to " + url + Fore.RESET)

if all_good:
    print_cond(log_verbose, Fore.GREEN + "All links to websites work fine !" + Fore.RESET)


# Get all jupyter notebooks included in the README
start = '\('
end = '.ipynb'

# Find the name of the tutorial
list_nb = re.findall('%s(.*)%s' % (start, end), data)
# Add ipynb extension to the filenames
list_nb = [x + ".ipynb" for x in list_nb]

# Check the notebooks links
print_cond(log_verbose, Fore.CYAN + "Checking notebooks URLs status code ... " + Fore.RESET)

raw_nb_url = []
all_good = True
for nb in list_nb:
    # Check that the notebook is not already given as a complete url
    if nb.find("http") == -1:
        url = "https://github.com/GUDHI/TDA-tutorial/blob/master/"+nb
        try:
            r = requests.head(url)
            if (r.status_code != 200):
                all_good = False
                if (r.status_code == 404):
                    print(Fore.RED + "{} is not working. The returned status code is {:4d}".format(url, r.status_code) + Fore.RESET)
                    exit_status = 1
                else:
                    print_cond(log_verbose, Fore.LIGHTYELLOW_EX + "{} may not be working. The returned status code is {:4d}".format(url, r.status_code) + Fore.RESET)
            else:
                raw_nb_url.append("https://raw.githubusercontent.com/GUDHI/TDA-tutorial/master/"+nb)
        except requests.ConnectionError:
            print_cond(log_verbose, Fore.RED + "Failed to connect to " + url + Fore.RESET)

if all_good:
    print_cond(log_verbose, Fore.GREEN + "All links to notebooks work fine !" + Fore.RESET)

# Check links inside notebooks
for nb in raw_nb_url:
    all_good = True
    print_cond(log_verbose, Fore.CYAN + "Checking URLs status code of notebook " + nb + " ..." + Fore.RESET)
    raw_nb_data  = urlopen(nb).read().decode('utf-8')
    # Beginning with http or https and ending with ' ', '"' or ')' or '\n' or ''' or '>'
    url_nb_regex = r"http[s]?://[^)\"\ \\\n\'\>]+"

    for match in re.findall(url_nb_regex, raw_nb_data):
        # Remove '.' if it was included in the url (by writer to end the sentence but is not supposed to be there)
        if (match.endswith('.')):
            match = match[:-1]
        # Irrelevant urls; could also add svg etc
        if ( match.endswith('.png') | (match.find("mapbox") != -1) ):
            continue
        # Check that links are not broken
        try:
            r = requests.head(match)
            if (r.status_code != 200):
                all_good = False
                if (r.status_code == 404):
                    print(Fore.RED + "{} is not working. The returned status code is {:4d}".format(match, r.status_code) + Fore.RESET)
                    exit_status = 1
                else:
                    print_cond(log_verbose, Fore.LIGHTYELLOW_EX + "{} may not be working. The returned status code is {:4d}".format(match, r.status_code) + Fore.RESET)
        except requests.ConnectionError:
            print_cond(log_verbose, Fore.RED + "Failed to connect to " + match + Fore.RESET)
    if all_good:
        print_cond(log_verbose, Fore.GREEN + "All links in the notebook work fine !" + Fore.RESET)

exit(exit_status)
