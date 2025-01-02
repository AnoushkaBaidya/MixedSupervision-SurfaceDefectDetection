from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import os

if __name__ == "__main__":
    print(" Start Downloading")

    zipurl = 'http://go.vicos.si/kolektorsdd2'

    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall('.')

    print("Extracting to:", os.getcwd())