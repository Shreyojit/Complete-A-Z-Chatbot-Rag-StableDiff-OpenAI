# src/utils/web_servers/rag_reference_service.py
import http.server
import socketserver
import yaml
import os
from pyprojroot import here
from load_web_service_config import LoadWebServicesConfig

WEB_SERVICE_CFG = LoadWebServicesConfig()

with open(here("configs/rag_gpt.yml")) as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)

DIRECTORY1 = app_config["directories"]["data_directory"]
DIRECTORY2 = app_config["directories"]["data_directory_2"]

class SingleDirectoryHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY1, **kwargs)

class MultiDirectoryHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        parts = path.split('/', 2)
        if len(parts) > 1:
            first_directory = parts[1]
            if first_directory == os.path.basename(DIRECTORY1):
                path = os.path.join(DIRECTORY1, *parts[2:])
                print(path)
            elif first_directory == os.path.basename(DIRECTORY2):
                path = os.path.join(DIRECTORY2, *parts[2:])
                print(path)
            else:
                file_path1 = os.path.join(DIRECTORY1, first_directory)
                print("file_path1", file_path1)
                file_path2 = os.path.join(DIRECTORY2, first_directory)
                print("file_path2", file_path2)
                if os.path.isfile(file_path1):
                    return file_path1
                elif os.path.isfile(file_path2):
                    return file_path2
        return super().translate_path(path)

if __name__ == "__main__":
    with socketserver.TCPServer(("", WEB_SERVICE_CFG.rag_reference_service_port), MultiDirectoryHTTPRequestHandler) as httpd:
        print(f"Serving at port {WEB_SERVICE_CFG.rag_reference_service_port}")
        httpd.serve_forever()
