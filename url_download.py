from urllib.request import urlopen, Request
from urllib.parse import urlsplit, urljoin
from urllib.error import *
import requests

def url_download(domain, file_sequence, EMAIL, AUTHKEY):
    downloaded = []
    error_others = []
    error_url = []

    # add '/' to domain if needed
    domain = parse_domain(domain)

    for file in file_sequence:
        url = urljoin(domain, file)
        print('url:\t{}'.format(url))
        try:
            try:
                contents = urlopen(url).read()
            except HTTPError:
                # authorization
                with requests.Session() as s:
                    s.auth = (EMAIL, AUTHKEY)
                    auth = s.post(url)
                    r = s.get(url)
                    contents = r.content
        except URLError:
            error_url.append(url)
        else:
            if contents:
                downloaded.append(file)
                with open(file, 'wb') as f:
                    f.write(contents)
            else:
                error_others.append(url)

    # log results in stdout
    if downloaded:
        print_url_message(downloaded, 'Downloaded')
    else:
        print('No files downloaded\n')
    print_url_message(error_others, 'Unable to get contents')
    print_url_message(error_url, 'URLError')

def print_url_message(lst, msg):
    '''
    :para lst: list of URLs
    :para msg: string, message to be printed

    prints message, and then prints all URLs that caused this message
    '''
    if lst:
        print(msg + ' for these URLs:')
        for url in lst:
            print(url)
        print()

def parse_domain(domain):
    '''
    :para domain: string, URL that links to a directory

    add forward slash to domain if does not exist at end of domain
    '''
    if domain[-1] != '/' and domain[-1] != '\\':
        domain += '/'
    return domain

def get_file_sequence(filename_filetype, start, end):
    '''
    :para domain: string, URL that links to a directory
    :para filename_filetype: dictionary, key is string of filename, value is string of filetype
    :para start: start of file numbering
    :para end: end of file numbering

    concatenates filename, file numbering, and filetype together into a full filename,
    where file numbering ranges from start to end.
    assumes file numbering is in range 01 to 99.

    returns list of full filenames, in sequence of file numbering
    '''
    if not 1 <= start <= end <= 99:
        raise Exception('File numbering wrong. Please ensure 1 <= start <= end <= 99')
    file_sequence = []
    for filename, filetype in filename_filetype.items():
        for i in range(start, end + 1):
            i = str(i)
            # pad
            if len(i) == 1:
                i = '0' + i
            file_sequence.append(filename + i + filetype)
    return file_sequence

if __name__ == '__main__':
    domain = 'http://challenges.tmlc1.unpossib.ly/api/datasets/'
    filename_filetype = {'tmlc1-training-':'.tar.gz', 'tmlc1-testing-':'.tar.gz'}
    EMAIL = 'zhengqun.koo@gmail.com'
    AUTHKEY = '69072a84e36a942c33a3ff678b6f23a4'
    file_sequence = get_file_sequence(filename_filetype, 1, 30)
    url_download(domain, file_sequence, EMAIL, AUTHKEY)