import sys

#app's path
sys.path.insert(0,"D:\\tf_server")

from tf_server import app

#Initialize WSGI app object
application = app