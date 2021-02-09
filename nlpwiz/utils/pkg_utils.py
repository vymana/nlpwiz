
import sys
import subprocess


def get_installed_packages():
    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
    return installed_packages


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


installed_packages = get_installed_packages()


def ensure_dependencies(packages):
    global installed_packages
    installed = False
    for package in packages:
        try:
            if package not in installed_packages:
                install(package)
                installed = True
        except Exception as ex:
            print("failed to install package: {}".format(package))
            print(str(ex))

    if installed:
        installed_packages = get_installed_packages()
