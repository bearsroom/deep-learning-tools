
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import os
import sys
import argparse
import random

im_template = '<div style="float:left; width:300px; height:300px;"> <a href="%s"> <img src="%s" alt="%s" height="256" width="256" style="border:1px solid red"> <div style=\'word-wrap: break-word; width: 250px; text-align: center;\'> %s </div> </a> </div>'
LIMIT = 0
MIMG_SERVER = ['10.2.96.34:10200', '10.2.96.35:10200']
MIMG_URL = 'http://%s/images/%s?m=2&w=600&h=600'
USE_MIMG = False

class ImageRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global http_template, header, im_template
        path = self.path.lstrip('/')
        if os.path.isdir(path):
            images = [im for im in os.listdir(path) if im.split('.')[-1].lower() in ('jpg', 'jpeg', 'png')]
            if images:
                self.send_response(200, 'OK')
                self.send_header('Content-tpye', 'text/html')
                self.end_headers()
                self.wfile.write(bytes('<html> <head><title> %s </title> </head> <body>' % path))
                cwd = os.getcwd()
                self.wfile.write('<br> Current dir: %s <br>' % (cwd+'/'+path))
                self.wfile.write('<br> Root dir: %s <br>' % cwd)
                self.wfile.write('<br> {} images in this dir <br> <br><br>'.format(len(images)))

                # render images in this dir
                im_render_list = []
                if LIMIT > 0:
                    images = images[:LIMIT]
                for im in images:
                    im_path = os.path.join(path, im)
                    im_href = MIMG_URL % (random.choice(MIMG_SERVER), im.split('.')[0]) if USE_MIMG else im_path
                    self.wfile.write(bytes(im_template % (im_href, im_path, im, im)))

                self.wfile.write(bytes(' </body> </html>'))
            else:
                dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                files = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
                if dirs or files:
                    self.send_response(200, 'OK')
                    self.send_header('Content-tpye', 'image/jpg')
                    self.end_headers()
                    for d in dirs:
                        link = '<br> <a href="/%s"> %s </a> <br>' % (path+'/'+d, path+'/'+d)
                        self.wfile.write(bytes(link))
                    for f in files:
                        line = '<br> <a href=""> %s </a> <br>' % path+'/'+f
                        self.wfile.write(bytes(line))
                else:
                    print('Code 404: no images in dir {}'.format(path))
                    content = 'Invalid Request: no images in dir {}'.format(path)
                    self.send_header('Content-tpye', 'text/html')
                    self.end_headers()
                    self.wfile.write(content)
        else:
            if os.path.isfile(path) and path.split('.')[-1].lower() in ('jpg', 'jpeg', 'png'):
                self.send_response(200, 'OK')
                self.send_header('Content-tpye', 'image/jpg')
                self.end_headers()
                self.wfile.write(open(path).read())

            elif path == '':
                self.send_response(200, 'OK')
                self.send_header('Content-tpye', 'text/html')
                self.end_headers()
                self.wfile.write(bytes('<html> <head><title> Root </title> </head> <body>'))
                cwd = os.getcwd()
                self.wfile.write('<br> Current dir: %s <br>' % (cwd+'/'+path))
                self.wfile.write('<br> Root dir: %s <br>' % cwd)
                self.wfile.write('<br> <br>')
                paths = os.listdir(cwd)
                for p in paths:
                    link = '<br> <a href="/%s"> %s </a> <br>' % (p, p)
                    self.wfile.write(bytes(link))

                self.wfile.write(bytes(' </body> </html>'))
            else:
                if not os.path.isfile(path):
                    print('Code 404: no such file {}'.format(path))
                else:
                    print('Code 404: cannot load this file as image: {}'.format(path))
                content = 'Invalid Request: no such path'
                return_link = '<br> <a href="//"> return to root dir: {} </a> <br>'.format(os.getcwd())
                self.send_header('Content-tpye', 'text/html')
                self.end_headers()
                self.wfile.write(bytes('<html> <head><title> Oops </title> </head> <body>'))
                self.wfile.write(bytes(content))
                self.wfile.write(bytes(return_link))
                self.wfile.write(bytes(' </body> </html>'))


class ImageHTTPServer(HTTPServer):
    def __init__(self, addr, handler, init_path=''):
        HTTPServer.__init__(self, addr, handler)
        if init_path != '' and os.path.isdir(init_path):
            self.chdir(init_path)

    def chdir(self, path):
        if os.path.isdir(path):
            os.chdir(path)
            print('Change dir to {}'.format(path))


def render(init_path='', addr='0.0.0.0', port=8080):
    try:
        Handler = ImageRequestHandler
        server = ImageHTTPServer((addr, port), Handler)
        server.chdir(init_path)

        print('Start server, addr: %s:%d' % (addr, port))
        server.serve_forever()
    except KeyboardInterrupt:
        print('Received exit signal from keyboard input')
        server.socket.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Simply render web page of images')
    parser.add_argument('--dir', required=True, help='Data dir to render')
    parser.add_argument('--addr', type=str, help='IP addr of server', default='0.0.0.0')
    parser.add_argument('--port', type=int, help='Port', default=8080)
    parser.add_argument('--limit', type=int, help='Image limit per page', default=0)
    parser.add_argument('--use-mimg', help='Use Mimg server to show image, only for online images', action='store_true')

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    LIMIT = args.limit
    USE_MIMG = args.use_mimg
    render(init_path=args.dir, addr=args.addr, port=args.port)



